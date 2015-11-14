require 'torch'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

local Dataset = torch.class('Dataset')
local TrainDs, parentTr = torch.class('TrainDs', 'Dataset')
local TestDs, parentTest = torch.class('TestDs', 'Dataset')


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
  local labels = f:read("labels"):partial({1, o_rows})
  local features = f:read("features"):partial({1, o_rows}, {1, o_cols})
  local o_data = {["features"]=features, ["labels"]=labels}
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
      if limit and  rows + size > limit then
        rows = limit - size
        l_reached = true
      end
      local cols = item.cols[1]
      for i=0, (rows - 1) do
        size = size + 1
        features[size] = item.data[{{i*cols + 1, (i+1)*cols}}]
      end
      table.insert(tags, {tag, rows})
    end      
  end 
  local o_data = {["features"]=features}
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
  for i=1, #tags do
    local tag = tags[i][1]
    local size = tags[i][2]
    local t_data = data[{{counter, counter + size - 1},{}}]
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



