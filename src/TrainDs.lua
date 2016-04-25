require 'torch'
require 'hdf5'
require 'utils'

torch.setdefaulttensortype('torch.FloatTensor')

-- Dataset for training purposes

local TrainDs = torch.class('TrainDs')


local SEQ_SIZES = "seq_sizes"
local LAB_SIZES = "lab_sizes"
local LABELS = "labels"
local FEATURES = "features"
local ROWS = "rows"
local COLS = "cols"

TrainDs.FRAME_ITER = "frame"
TrainDs.SEQ_ITER = "seq"
TrainDs.FRAC_ITER = "seq_frac"


function TrainDs:__init(filename, cuda, load_all)
    self.filename = filename
    self.load_all = load_all
    self.f = hdf5.open(self.filename)
    self.f_labels = self.f:read(LABELS)
    self.f_features = self.f:read(FEATURES)
    self.data = torch.Tensor()
    self.labels = torch.Tensor()
    self:_readSize()
    self:_readSeqSizes()
    if load_all then
        self:_readAll()
    end
    self:_genIntervals()
end

function TrainDs:_readAll()
    self.a_features = self.f_features:all()
    self.a_labels = self.f_labels:all():type('torch.FloatTensor')
end

function TrainDs:_readSeqSizes()
    local status, seq_sizes = pcall(self.f.read, self.f, SEQ_SIZES)
    if status then
        self.seq_sizes = seq_sizes:all()[1]
        local status, lab_sizes = pcall(self.f.read, self.f, LAB_SIZES)
        -- if label sizes are not avalaible assume that lables has the same
        -- length as the input sequences
        if status then
            self.lab_sizes = lab_sizes:all()[1]
        else
            self.lab_sizes = self.seq_sizes
        end
        assert(self.lab_sizes:size(1) == self.seq_sizes:size(1),
            "There is not corresponding amount of sequences and labels.")
    else
        error("Could not read sequence sizes.")
    end
end

function TrainDs:_genIntervals()
    self.seqIntervals = {}
    local acc = 0
    for i = 1, self.seq_sizes:size(1) do
        acc = acc + self.seq_sizes[i]
        table.insert(self.seqIntervals, acc)
    end

    self.labIntervals = {}
    for i = 1, self.lab_sizes:size(1) do
        acc = acc + self.lab_sizes[i]
        table.insert(self.labIntervals, acc)
    end
end


function TrainDs:_readSize()
    local s_rows, rows = pcall(self.f.read, self.f, ROWS)
    local s_cols, cols = pcall(self.f.read, self.f, COLS)
    local rows = rows:all()
    local cols = cols:all()
    if s_rows and s_cols then
        self.rows = rows[1]
        self.cols = cols[1]
    else
        error("Could not read size of the dataset.")
    end
end

function TrainDs:_getData(dataInt, labInt)
    labInt = labInt or dataInt
    local data = nil
    local labels = nil
    if self.load_all then
        data = self.a_features[{ dataInt, { 1, self.cols } }]:clone()
        labels = self.a_labels[{ labInt }]:clone()
    else
        data = self.f_features:partial(labInt, { 1, self.cols })
        labels = self.f_labels:partial(labInt)
    end
    labels:add(1) -- lua indexes from 1
    return data, labels
end


function TrainDs:startSeqIteration(shuffle)
    self.seq_index = 1
    if shuffle then
        self.seq_indexes = torch.randperm(#self.seqIntervals)
    else
        self.seq_indexes = torch.range(1, #self.seqIntervals)
    end
end


function TrainDs:startFracIteration(shuffle)
    self.seq_index = 1
    local bCount = #self.seqIntervals / self.bSize
    if #self.seqIntervals % self.bSize ~= 0 then
        bCount = bCount + 1
    end
    local fracIndexes
    if shuffle then
        fracIndexes = torch.randperm(bCount)
    else
        fracIndexes = torch.range(1, bCount)
    end
    self.seq_indexes = torch.zeros(#self.seqIntervals)
    local index = 1
    for i=1, bCount do
        local s = (fracIndexes[i] - 1) * self.bSize + 1
        local e = math.min(s + self.bSize - 1, #self.seqIntervals)
        for y=s, e do
            self.seq_indexes[index] = y
            index = index + 1
        end
    end
end


function TrainDs:getSeq()
    if self.seq_index > #self.seqIntervals then
        return nil
    end
    local r_index = self.seq_indexes[self.seq_index]
    local startSeq = 1
    local startLab = 1
    if r_index ~= 1 then
        startSeq = self.seqIntervals[r_index - 1] + 1
        startLab = self.labIntervals[r_index - 1] + 1
    end
    local seqInterval = { startSeq, self.seqIntervals[r_index] }
    local labInterval = { startSeq, self.labIntervals[r_index] }
    self.seq_index = self.seq_index + 1
    return self:_getData(seqInterval, labInterval)
end


function TrainDs:startFrameIteration(shuffle)
    assert(self.a_features:size(1) == self.a_labels:size(1),
        "Feature and label counts do not match.")
    self.fIndex = 1
    if shuffle then
        self.fIndexes = torch.randperm(self.a_features:size(1))
    else
        self.fIndexes = torch.range(1, self.a_features:size(1))
    end
end


function TrainDs:getFrame()
    if self.fIndex > self.fIndexes:size(1) then
        return nil
    end
    local index = self.fIndexes[self.fIndex]
    local interval = { index, index }
    local data, labels = self:_getData(interval, interval)
    self.fIndex = self.fIndex + 1
    return data, labels
end


function TrainDs:startBatchIteration(type, bSize, shuffle, hSize, split,
                                     formatLabels)
    self.bSize = bSize
    self.hSize = hSize or math.huge
    self.shuffle = shuffle
    self.formatLabels = formatLabels
    self.split = split
    self.buffer = {}
    self.sizes = {}
    if type == TrainDs.FRAME_ITER then
        self.getItem = self.getFrame
        self:startFrameIteration(shuffle)
    elseif type == TrainDs.SEQ_ITER then
        self.getItem = self.getSeq
        self:startSeqIteration(shuffle)
    elseif type == TrainDs.FRAC_ITER then
        self.getItem = self.getSeq
        self:startFracIteration(shuffle)
    else
        error("Unknown iteration type.")
    end
end


function TrainDs:nextBatch()
    while #self.buffer < self.bSize do
        local data, labels
        data, labels = self:getItem()
        -- no more data to process
        if data == nil then
            break
        end
        if data:size(1) <= self.hSize then
            table.insert(self.buffer, { data:clone(), labels:clone() })
            table.insert(self.sizes, data:size(1))
            -- ignore/split sequences that are longer than actual hSize
        elseif self.split then
            assert(data:size(1) == labels:size(1),
                "Splitting is allowed only if labels and data have the same size.")
            local index = 0
            local int
            for i = 1, data:size(1) - self.hSize, self.hSize do
                int = { { index + 1, index + self.hSize } }
                table.insert(self.buffer, { data[int]:clone(), labels[int]:clone() })
                table.insert(self.sizes, self.hSize)
                index = index + self.hSize
            end
            int = { { data:size(1) - self.hSize + 1, data:size(1) } }
            table.insert(self.buffer, { data[int]:clone(), labels[int]:clone() })
            table.insert(self.sizes, self.hSize)
        end
    end

    if #self.buffer == 0 then
        return nil
    end

    table.sort(self.buffer, function(a, b) return a[1]:size(1) > b[1]:size(1) end)
    table.sort(self.sizes, function(a, b) return a > b end)
    local iCount = 0
    local aBCount = math.min(self.bSize, #self.buffer)
    for i = 1, aBCount do
        iCount = iCount + self.sizes[i]
    end
    self.data:resize(iCount, self.buffer[1][1]:size(2))
    local maxT = self.sizes[1]
    local index = 1
    for t = 1, maxT do
        for i = 1, aBCount do
            if self.buffer[i][1]:size(1) >= t then
                self.data[index]:copy(self.buffer[i][1][t])
                index = index + 1
            end
        end
    end

    local labels = {}
    -- standard crossentropy
    if self.formatLabels then
        self.labels:resize(iCount)
        local index = 1
        for t = 1, maxT do
            for i = 1, aBCount do
                if self.buffer[i][2]:size(1) >= t then
                    self.labels[index] = self.buffer[i][2][t]
                    index = index + 1
                end
            end
        end
        labels = self.labels
    else -- ctc
    for i = 1, aBCount do
        table.insert(labels, self.buffer[i][2]:float():totable())
    end
    end
    -- remove used seqs
    local aSizes = {}
    for i = 1, aBCount do
        table.insert(aSizes, table.remove(self.sizes, 1))
        table.remove(self.buffer, 1)
    end
    return self.data, labels, aSizes
end


--eof
