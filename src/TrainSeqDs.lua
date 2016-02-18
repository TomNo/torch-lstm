require 'torch'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

-- Sequential dataset used for sequential training


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
    self.a_labels = self.f_labels:all()
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

function TrainSeqDs:_getSeq(interval)
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
    local data, labels = self:_getSeq(interval)
    self.data:resize(data:size(1), data:size(2))
    self.labels:resize(labels:size(1))
    self.data:copy(data)
    self.labels:copy(labels)
    return self.data, self.labels
end


function TrainSeqDs:_prepareSeqData(restDataIndex)
    local seq, labels = self:getSeq()
    if not seq then
        return false
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
    if self.overlap then
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

function TrainSeqDs:startBatchIteration(b_size, h_size, shuffle, overlap, rShift)
    self:startSeqIteration(shuffle)
    self.a_seq_index = 1
    self.h_size = h_size
    self.b_size = b_size
    self.b_count = h_size * b_size
    self:_prepareSeqData()
    self.overlap = overlap or false
    self.rShilt = rShift or false
end

function TrainSeqDs:nextBatch()
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
    self.data:copy(tmp_data:reshape(self.b_size, self.h_size, self.cols):transpose(1, 2):reshape(self.b_count, self.cols))
    self.labels:copy(tmp_labels:reshape(self.b_size, self.h_size):t():reshape(self.b_count))

    self.a_seq_index = self.a_seq_index + self.b_count
    return self.data, self.labels
end


--eof
