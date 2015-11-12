require 'torch'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

local Dataset = torch.class('Dataset')

function Dataset:__init(filename)
  self.filename = filename
end

-- iterator over hdf5
function Dataset:dataIter(f)
  local counter = 0
  return function()
           local status, data = pcall(f.read, f, "/" .. counter)
           if status then
             counter = counter + 1
             return data:all()
           end
         end
end

-- goes through the whole dataset(if limit not specified) and gather number of rows and columns
function Dataset:getSize(f, limit)
  local iter = self:dataIter(f, limit)
  local rows = 0
  local cols = 0
  local first = true
  while true do
    local item = iter()
    if not item then
      break
    end
    for _, d_item in pairs(item) do
      if first then
        first = false
        cols = d_item.cols[1]
      end
      rows = rows + d_item.rows[1]
      if limit and rows > limit then
        return limit, cols
      end    
    end
  end
  return rows, cols
end

-- Checks wheather dataset cotains labels
function Dataset:labelsPresent(f)
  if self.l_present ~= nil then
    return self.l_present
  end
  local iter = self:dataIter(f)
  for _, item in pairs(iter()) do
    if item.labels then
      self.l_present = true
    else
      self.l_present = false
    end
    break
  end
  return self.l_present
end

-- reads the data from input file and returns the dataset
function Dataset:get(limit)
  if limit then
    print("Loading dataset with limit " .. limit .. " from file " .. self.filename)
  else
    print("Loading dataset from file " .. self.filename)
  end
  local f = hdf5.open(self.filename)
  local o_data = {}
  local o_rows = 0
  local o_cols = 0
  o_rows, o_cols = self:getSize(f, limit)
  local features = torch.Tensor(o_rows, o_cols)
  local labels = nil
  if self:labelsPresent(f) then
    labels = torch.Tensor(o_rows)  
  end
  local tags = {}
  local size = 0
  local l_reached = false
  local d_iter = self:dataIter(f)
  while true do
    if l_reached then
      break
    end
    local d_item = d_iter()
    if not d_item then
      break
    end
    for tag, item in pairs(d_item) do
      print("Processing tag " .. tag)
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
      if self:labelsPresent(f) then
        labels[{{size - rows + 1, size}}] = item.labels
      end
      table.insert(tags, {tag, rows})
    end      
  end 
  o_data.size = function () return o_rows end
  o_data.tags = tags
  o_data = {[1]=features, [2]=labels}
  print("Dataset loaded.")
  f:close()
  return o_data
end

-- saves the data into output file specified on initialization
function Dataset:save(data, tags)
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
    f:write(tag .. "/data", t_data:view(t_data:nElement()))
    f:write(tag .. "/rows", rows)
    f:write(tag .. "/cols", cols)
    counter = counter + size
  end
  f:close()
  print("Dataset saved.")
end