require 'torch'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

-- Sequential dataset used for sequential training

--TODO this must be refactored, some iteration class should be created

local TrainSeqDs = torch.class('TrainSeqDs')


local SEQ_SIZES = "seq_sizes"
local LABELS = "labels"
local FEATURES = "features"
local ROWS = "rows"
local COLS = "cols"


function TrainSeqDs:__init(filename, cuda, load_all)
    self.filename = filename
    self.load_all = load_all
    self.f = hdf5.open(self.filename)
    self.f_labels = self.f:read(LABELS)
    self.f_features = self.f:read(FEATURES)
    if cuda then
        self.data = torch.CudaTensor()
        self.labels = torch.CudaTensor()
    else
        self.data = torch.Tensor()
        self.labels = torch.Tensor()
    end
    self:_readSize()
    self:_readSeqSizes()
    if load_all then
        self:_readAll()
    end
    self:_genIntervals()
end

function TrainSeqDs:_readAll()
    self.a_features = self.f_features:all()
    self.a_labels = self.f_labels:all():type('torch.FloatTensor')
end

function TrainSeqDs:_readSeqSizes()
    local status, seq_sizes = pcall(self.f.read, self.f, SEQ_SIZES)
    if status then
        self.seq_sizes = seq_sizes:all()[1]
    else
        error("Could not read sequence sizes.")
    end
end

function TrainSeqDs:_genIntervals()
    self.intervals = {}
    local acc = 0
    for i = 1, self.seq_sizes:size(1) do
        acc = acc + self.seq_sizes[i]
        table.insert(self.intervals, acc)
    end
end

function TrainSeqDs:_genFracIntervals()
    self.fracIntervals = {}
    local acc = 1
    for i = 1, #self.intervals do
        local interval = self.intervals[i]
        local shift = self.h_size
        if self.overlap then
            shift = math.floor(self.h_size / 2)
        end
        -- ignoring shorter sequences than is history
        if interval - acc + 1 >= self.h_size then
            for y=acc, interval - self.h_size, shift do
                local e = y + self.h_size - 1
                table.insert(self.fracIntervals, {y, e})
            end
            -- insert stuff that did not make minibatch regularly
            table.insert(self.fracIntervals, {interval - self.h_size + 1,
                                              interval})
        end
        acc = 1 + interval -- start of the future frame
    end
end

function TrainSeqDs:_readSize()
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

function TrainSeqDs:_getData(interval)
    local data = nil
    local labels = nil
    if self.load_all then
        data = self.a_features[{ interval, { 1, self.cols } }]:clone()
        labels = self.a_labels[{ interval }]:clone()
    else
        data = self.f_features:partial(interval, { 1, self.cols })
        labels = self.f_labels:partial(interval)
    end
    labels:add(1) -- lua indexes from 1
    return data, labels
end

function TrainSeqDs:startSeqIteration(shuffle)
    self.seq_index = 1
    if shuffle then
        self.seq_indexes = torch.randperm(#self.intervals)
    else
        self.seq_indexes = torch.range(1, #self.intervals)
    end
end

function TrainSeqDs:getSeq()
    if self.seq_index > #self.intervals then
        return nil
    end
    local r_index = self.seq_indexes[self.seq_index]
    local start = 1
    if r_index ~= 1 then
        start = self.intervals[r_index - 1] + 1
    end
    local interval = { start, self.intervals[r_index] }
    self.seq_index = self.seq_index + 1
    local data, labels = self:_getData(interval)
    self.data:resize(data:size(1), data:size(2))
    self.labels:resize(labels:size(1))
    self.data:copy(data)
    self.labels:copy(labels)
    return self.data, self.labels
end


function TrainSeqDs:startParallelSeq(b_size, h_size, shuffle, convertLabels)
    self:startSeqIteration(shuffle)
    self.h_size = h_size
    self.b_size = b_size
    self.b_count = h_size * b_size
    self.convertLabels = convertLabels or true
    self.nextBatch = self.nextParallelSeq
end


function TrainSeqDs:nextParallelSeq()
    local bufferSize = 0
    self.seqBuffer = {}
    self.labBuffer = {}
    self.sizes = {}
    local maxSize = 0
    while bufferSize < self.b_size do
        local data, labels
        data, labels = self:getSeq()
        -- no more data to process
        if data == nil then
            break
        end
        -- ignore sequences that are longer than actual h_size
        if data:size(1) <= self.h_size then
            table.insert(self.seqBuffer, {data:clone(), labels:clone()})
            table.insert(self.sizes, data:size(1))
            bufferSize = bufferSize + 1
            if data:size(1) > maxSize then
                maxSize = data:size(1)
            end
        end
    end

    if bufferSize == 0 then
        return nil
    end

    table.sort(self.seqBuffer, function(a, b) return a[1]:size(1) > b[1]:size(1) end)
    table.sort(self.sizes, function (a, b) return a > b end)
    self.data:resize(self.b_count, self.seqBuffer[1][1]:size(2))
    self.data:zero()
    for i=1, bufferSize do
        for y=1, self.seqBuffer[i][1]:size(1) do
            self.data[i + (y - 1) * self.b_size]:copy(self.seqBuffer[i][1][y])
        end
    end


    if self.convertLabels then
        self.labels:resize(self.b_count)
        self.labels:zero()
        for i=1, bufferSize do
            for y=1, self.seqBuffer[i][2]:size(1) do
                self.labels[i + (y - 1) * self.b_size] = self.seqBuffer[i][2][y]
            end
        end
    else
        self.labels = {}
        for i=1, bufferSize do
            table.insert(self.labels, self.seqBuffer[i][2])
        end
    end

    return self.data, self.labels, self.sizes
end



--function TrainSeqDs:_getFrac()
--    if self.fracIndex > #self.fracIntervals then
--        return nil
--    end
--    local index  = self.fracIndexes[self.fracIndex]
--    local data, labels = self:_getData(self.fracIntervals[index])
--    self.fracIndex = self.fracIndex + 1
--    return data, labels
--end
--
--
function TrainSeqDs:_prepareSeqData(restDataIndex)
    local seq, labels
    -- ami contains short sequences sometime....
    while not seq or seq:size(1) < self.h_size do
        seq, labels = self:getSeq()
        if not seq then
            return false
        end
    end
    -- do random shift of the sequence start so we get a little bit different
    -- sequences in mini batches each time
    if self.rShift then
        local shift = math.random(1, self.h_size - 1)
        seq = seq[{{shift, seq:size(1)}}]
        labels = labels[{{shift, labels:size(1)}}]
    end

    --duplicate data that does not form minibatch
    local m_count = math.floor(seq:size(1) / self.h_size)
    local overhang = seq:size(1) % self.h_size
    if overhang > 0 then
        seq:resize(self.h_size * (m_count + 1), seq:size(2))
        labels:resize(self.h_size * (m_count + 1))
        local e = self.h_size * m_count + overhang
        local s = e - self.h_size + 1
        local from = {{s, e }}
        local to = {{self.h_size * m_count + 1, seq:size(1) }}
        seq[to]:copy(seq[from]:clone())
        labels[to]:copy(labels[from]:clone())
        m_count = m_count + 1
    end

    local m_count = math.floor(seq:size(1) / self.h_size)
    local shift = (self.h_size / 2)
    local m_s_count = math.floor((seq:size(1) - shift) / self.h_size)
    local r_data, r_labels
    if restDataIndex ~= nil then
        local o_size = self.seq_data:size(1)
        r_data = self.seq_data[{ { restDataIndex, o_size } }]
        r_labels = self.seq_labels[{ { restDataIndex, o_size } }]
    end
    local int = { { 1, m_count * self.h_size } }
    self.seq_data = seq[int]
    self.seq_labels = labels[int]
    if self.overlap and m_s_count ~= 0 then
        local shift_int = { { shift, m_s_count * self.h_size } }
        self.seq_data = self.seq_data:cat(seq[shift_int], 1)
        self.seq_labels = self.seq_labels:cat(labels[shift_int], 1)
    end
    if restDataIndex ~= nil then
        self.seq_data = r_data:cat(self.seq_data, 1)
        self.seq_labels = r_labels:cat(self.seq_labels, 1)
    end
    return true
end

function TrainSeqDs:nextClassicBatch()
    while self.a_seq_index + self.b_count > self.seq_data:size(1) do
        local status = self:_prepareSeqData(self.a_seq_index)
        if status then
            self.a_seq_index = 1
        else
            return nil
        end
    end
    self.data:resize(self.b_count, self.cols)
    self.labels:resize(self.b_count)
    local int = { { self.a_seq_index, self.a_seq_index + self.b_count - 1 } }
    local tmp_data = self.seq_data[int]
    local tmp_labels = self.seq_labels[int]
    self.data:copy(tmp_data:view(self.b_size, self.h_size, self.cols):transpose(1, 2):reshape(self.b_count, self.cols))
    self.labels:copy(tmp_labels:view(self.b_size, self.h_size):t():reshape(self.b_count))

    self.a_seq_index = self.a_seq_index + self.b_count
    self.sizes = {}
    for i=1, self.b_count/self.h_size do
        table.insert(self.sizes, self.h_size)
    end

    return self.data, self.labels, self.sizes
end


function TrainSeqDs:startBatchIteration(b_size, h_size, shuffle, rShift,
                                        overlap)
    self.a_seq_index = 1
    self:startSeqIteration(shuffle)
    self.h_size = h_size
    self.b_size = b_size
    self.b_count = h_size * b_size
    self:_prepareSeqData()
    self.overlap = overlap or false
    self.rShilt = rShift or false
    self.nextBatch = self.nextClassicBatch
end
--
--function TrainSeqDs:startFractionIteration(shuffle)
--    self.fracIndex = 1
--    self:_genFracIntervals()
--    if shuffle then
--        self.fracIndexes = torch.randperm(#self.fracIntervals)
--    else
--        self.fracIndexes = torch.range(1, #self.fracIntervals)
--    end
--end
--
--function TrainSeqDs:startBatchIterationFractions(b_size, h_size, shuffle, rShift,
--                                                 overlap)
--    self.a_seq_index = 1
--    self.h_size = h_size
--    self.b_size = b_size
--    self.b_count = h_size * b_size
--    self.overlap = overlap or false
--    self.rShilt = rShift or false
--    self:startFractionIteration(shuffle)
--end
--
--function TrainSeqDs:startBatchIterationParallel(b_size, h_size, shuffle, rShift,
--                                        overlap)
--    self.a_seq_index = 1
--    self:startSeqIteration(shuffle)
--    self.h_size = h_size
--    self.b_size = b_size
--    self.b_count = h_size * b_size
----    self:_prepareSeqData()
--    self.overlap = overlap or false
--    self.rShilt = rShift or false
--    self.seqBuffer = {}
--    for i=1, b_size do
--        table.insert(self.seqBuffer, {})
--    end
--end
--
--function TrainSeqDs:_refillSeqBuffer()
--    for i=1, self.b_size do
--        local item = self.seqBuffer[i]
--        if not item.seq then
--            local seq, labels
--            repeat
--                seq, labels = self:getSeq()
--                if not seq then
--                    return false
--                end
--                -- ignore sequences that are shorter than history
--            until seq:size(1) >= self.h_size
--            --duplicate data that does not form minibatch
--            local m_count = math.floor(seq:size(1) / self.h_size)
--            local overhang = seq:size(1) % self.h_size
--            if overhang > 0 then
--                seq:resize(self.h_size * (m_count + 1), seq:size(2))
--                labels:resize(self.h_size * (m_count + 1))
--                local e = self.h_size * m_count + overhang
--                local s = e - self.h_size + 1
--                local from = {{s, e }}
--                local to = {{self.h_size * m_count + 1, seq:size(1) }}
--                seq[to]:copy(seq[from]:clone())
--                labels[to]:copy(labels[from]:clone())
--                m_count = m_count + 1
--            end
--            item.seq = seq:clone()
--            item.labels = labels:clone()
--        end
--    end
--    return true
--end
--
--
--function TrainSeqDs:nextParallelBatch()
--    -- refill only if buffer is empty
--    local emptySeqs = 0
--    for i=1, self.b_size do
--        if not self.seqBuffer[i].seq then
--            emptySeqs = emptySeqs + 1
--        end
--    end
--
--    if emptySeqs == self.b_size then
--        self:_refillSeqBuffer()
--    end
--
--    -- checks that there are still some sequeces to process
--    local seqCount = 0
--    for i=1, self.b_size do
--        local item = self.seqBuffer[i]
--        if item.seq then
--            seqCount = seqCount + 1
--        end
--    end
--
--    if seqCount == 0 then
--        return nil
--    end
--    self.data:resize(seqCount*self.h_size, self.cols)
--    self.labels:resize(seqCount* self.h_size)
--    local dIndex = 0
--    for y=1, self.h_size do
--        for i=1, self.b_size do
--            local item = self.seqBuffer[i]
--            if item.seq then
--                dIndex = dIndex + 1
--                self.data[dIndex] = item.seq[1]
--                self.labels[dIndex] = item.labels[1]
--                if item.seq:size(1) == 1 then
--                    item.seq = nil
--                    item.labels = nil
--                else
--                    item.seq = item.seq[{{2, item.seq:size(1)}}]
--                    item.labels = item.labels[{{2, item.labels:size(1)}}]
--                end
--            end
--        end
--    end
--
----    local int = { { self.a_seq_index, self.a_seq_index + self.b_count - 1 } }
----    local tmp_data = self.seq_data[int]
----    local tmp_labels = self.seq_labels[int]
----    self.data:copy(tmp_data:view(self.b_size, self.h_size, self.cols):transpose(1, 2):reshape(self.b_count, self.cols))
----    self.labels:copy(tmp_labels:view(self.b_size, self.h_size):t():reshape(self.b_count))
----
----    self.a_seq_index = self.a_seq_index + self.b_count
--    return self.data, self.labels
--end
--
--function TrainSeqDs:nextFracBatch()
--    local aBatchSize = 0
--    local bData = {}
--    local bLabels = {}
--    for i=1, self.b_size do
--        local data, labels = self:_getFrac()
--        if not data then
--            break
--        end
--        aBatchSize = i
--        table.insert(bData, data)
--        table.insert(bLabels, labels)
--    end
--    if aBatchSize == 0 then
--        return nil
--    end
--
--    self.data:resize(aBatchSize * self.h_size, self.cols)
--    self.labels:resize(aBatchSize * self.h_size)
--    local index = 1
--    for i=1, self.h_size do
--        for y=1, aBatchSize do
--            self.data[index] = bData[y][i]
--            self.labels[index] = bLabels[y][i]
--            index = index + 1
--        end
--    end
--    return self.data, self.labels
--end
--

--eof
