require 'torch'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

local Dataset = torch.class('Dataset')
local TrainDs, parentTr = torch.class('TrainDs', 'Dataset')
local TestDs, parentTest = torch.class('TestDs', 'Dataset')
local TrainSeqDs = torch.class('TrainSeqDs')

local SEQ_SIZES = "seq_sizes"
local LABELS = "labels"
local FEATURES = "features"
local ROWS = "rows"
local COLS = "cols"

-- Sequential dataset used for training
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

--function TrainSeqDs:startBatchIterationa(b_size, h_size, shuffle)
--  self.m_b_count = 2 * math.floor(self.rows / (h_size * b_size)) - 1
--  if shuffle then
--    self.samples = torch.randperm(2*(math.floor(self.rows / h_size) - 1))
--  else
--    self.samples = torch.range(1, 2*(math.floor(self.rows / h_size) - 1))
--  end
--  self.index = 1
--  self.h_size = h_size
--  self.b_size = b_size
--  self.a_b_count = 1
--end

function TrainSeqDs:_prepareSeqData(restDataIndex)
    local seq, labels = self:getSeq()
    if not seq then
        return false
    end
    local m_count = math.floor(seq:size(1) / self.h_size)
    local shift = (self.h_size / 2)
    local m_s_count = math.floor((seq:size(1) - shift) / self.h_size)
    local r_data = nil
    local r_labels = nil
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

function TrainSeqDs:startRealBatch(b_size, h_size, shuffle, overlap)
    self:startSeqIteration()
    self.a_seq_index = 1
    self.h_size = h_size
    self.b_size = b_size
    self.b_count = h_size * b_size
    self:_prepareSeqData()
    self.overlap = overlap
end

function TrainSeqDs:nextRealBatch()
    while self.a_seq_index + self.b_count > self.seq_data:size(1) do
        -- TODO something should be done with the data that does not form minibatch
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

function TrainSeqDs:startIteration(b_size, h_size, shuffle, train)
    local a = 1
    local b = 0
    -- in case of training make mini-sequences overlap
    if train then
        a = 2
        b = 1
    end
    self.m_b_count = a * math.floor(self.rows / (h_size * b_size)) - b
    if shuffle then
        self.samples = torch.randperm(a * (math.floor(self.rows / h_size)) - b)
    else
        self.samples = torch.range(1, a * (math.floor(self.rows / h_size)) - b)
    end
    self.index = 1
    self.h_size = h_size
    self.b_size = b_size
    self.a_b_count = 1
end

function TrainSeqDs:nextBatch()
    if self.a_b_count > self.m_b_count then
        return nil
    end
    self.data:resize(self.h_size * self.b_size, self.cols)
    self.labels:resize(self.b_size * self.h_size)
    for i = 1, self.b_size do
        local index = self.samples[self.index]
        local shift = 0
        if index >= (self.rows / self.h_size) then
            shift = self.h_size * 0.5
            index = math.floor(index / 2)
        end
        local interval = { (index - 1) * self.h_size + 1 + shift, index * self.h_size + shift }
        local seq_data, seq_labels = self:_getSeq(interval)
        for y = 0, self.h_size - 1 do
            self.labels[y * self.b_size + i] = seq_labels[y + 1]
            self.data[y * self.b_size + i] = seq_data[y + 1]
        end
        self.index = self.index + 1
    end
    self.a_b_count = self.a_b_count + 1
    return self.data, self.labels
end

-- Sequential training dataset end

function Dataset:__init(filename)
    self.filename = filename
end

-- gets rows and cols of given dataset
function Dataset:getSize(f, limit)
    --  local iter = self:dataIter(f, limit)
    local s_rows = false
    local s_cols = false
    local rows = 0
    local cols = 0
    s_rows, rows = pcall(f.read, f, "/rows")
    s_cols, cols = pcall(f.read, f, "/cols")
    rows = rows:all()
    cols = cols:all()
    if s_rows and s_cols then
        if limit and limit < rows[1] then
            return limit, cols[1]
        else
            return rows[1], cols[1]
        end
    else
        error("Could not read size of the dataset.")
    end
    --  rows = 0
    --  cols = 0
    --  local first = true
    --  while true do
    --    local item = iter()
    --    if not item then
    --      break
    --    end
    --    for _, d_item in pairs(item) do
    --      if first then
    --        first = false
    --        cols = d_item.cols[1]
    --      end
    --      rows = rows + d_item.rows[1]
    --      if limit and rows > limit then
    --        return limit, cols
    --      end
    --    end
    --  end
    --  return rows, cols
end

function TrainDs:__init(filename)
    parentTr.__init(self, filename)
end

function TrainDs:get(limit)
    if limit then
        print("Loading training dataset with limit " .. limit .. " from file " .. self.filename)
    else
        print("Loading training dataset from file " .. self.filename)
    end
    local f = hdf5.open(self.filename)
    local o_rows = 0
    local o_cols = 0
    o_rows, o_cols = self:getSize(f, limit)
    local labels = f:read("labels"):partial({ 1, o_rows })
    labels:add(1) -- lua indexing stars from 1
    local features = f:read("features"):partial({ 1, o_rows }, { 1, o_cols })
    local o_data = { ["features"] = features, ["labels"] = labels }
    o_data.size = function() return o_rows end
    o_data.rows = o_rows
    o_data.cols = o_cols
    print("Dataset loaded.")
    f:close()
    return o_data
end

function TestDs:__init(filename)
    parentTest.__init(self, filename)
end

-- iterator over test structure of hdf5
function TestDs:dataIter(f)
    local counter = 0
    return function()
        local status, data = pcall(f.read, f, "/" .. counter)
        if status then
            counter = counter + 1
            return data:all()
        end
    end
end

-- reads the data from input file and returns the dataset
function TestDs:get(limit)
    if limit then
        print("Loading test dataset with limit " .. limit .. " from file " .. self.filename)
    else
        print("Loading test dataset from file " .. self.filename)
    end
    local f = hdf5.open(self.filename)
    local o_rows = 0
    local o_cols = 0
    o_rows, o_cols = self:getSize(f, limit)
    local features = torch.Tensor(o_rows, o_cols)
    local tags = {}
    local size = 0
    local l_reached = false
    local d_iter = self:dataIter(f)
    while true do
        if l_reached then -- if limit reached -> stop
        break
        end
        local d_item = d_iter()
        if not d_item then -- if no more items -> stop
        break
        end
        for tag, item in pairs(d_item) do -- there is not other way to get that one key
        local rows = item.rows[1]
        if limit and rows + size > limit then
            rows = limit - size
            l_reached = true
        end
        local cols = item.cols[1]
        for i = 0, (rows - 1) do
            size = size + 1
            features[size] = item.data[{ { i * cols + 1, (i + 1) * cols } }]
        end
        table.insert(tags, { tag, rows })
        end
    end
    local o_data = { ["features"] = features }
    o_data.size = function() return o_rows end
    o_data.rows = o_rows
    o_data.cols = o_cols
    o_data.tags = tags
    print("Dataset loaded.")
    f:close()
    return o_data
end

-- saves the data into output file specified on initialization
function TestDs:save(data, tags)
    print("Saving dataset into the file " .. self.filename)
    local f = hdf5.open(self.filename, "w")
    local counter = 1
    for i = 1, #tags do
        local tag = tags[i][1]
        local size = tags[i][2]
        local t_data = data[{ { counter, counter + size - 1 }, {} }]
        local rows = torch.Tensor(1)
        rows[1] = t_data:size(1)
        local cols = torch.Tensor(1)
        cols[1] = t_data:size(2)
        f:write(tag .. "/data", t_data)
        f:write(tag .. "/rows", rows)
        f:write(tag .. "/cols", cols)
        counter = counter + size
    end
    f:close()
    print("Dataset saved.")
end



-- Checks wheather dataset cotains labels
--function Dataset:labelsPresent(f)
--  if self.l_present ~= nil then
--    return self.l_present
--  end
--  local iter = self:dataIter(f)
--  for _, item in pairs(iter()) do
--    if item.labels then
--      self.l_present = true
--    else
--      self.l_present = false
--    end
--    break
--  end
--  return self.l_present
--end



