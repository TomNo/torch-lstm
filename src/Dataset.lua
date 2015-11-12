require 'torch'
require 'hdf5'


local Dataset = torch.class('Dataset')

function Dataset:__init(filename)
  self.filename = filename
end

-- reads the data from input file and returns the dataset
-- TODO refactor
function Dataset:get(limit)
  if limit then
    print("Loading dataset with limit " .. limit .. " from file " .. self.filename)
  else
    print("Loading dataset from file " .. self.filename)
  end
  local f = hdf5.open(self.filename)
  local parsed_data = {}
  local tags = {}
  local size = 0
  local counter = 1
  local get_data = function() return f:read("/" .. counter):all() end
  while true do
    local status, t_data = pcall(get_data)
    if status then
      for tag, item in pairs(t_data) do
        print("Processing tag " .. tag)
        local cols = item.cols[1]
        local rows = item.rows[1]
        local data = item.data:reshape(rows, cols)
        local a_size = rows
        for i=1, rows do
          size = size + 1
          if item.labels then
            table.insert(parsed_data, {[1]=data[i], [2]=item.labels[{{i}}]})
          else
            table.insert(parsed_data, {[1]=data[i]})
          end
          if size + 1 == limit then
            a_size = i
            break
          end
        end
        table.insert(tags, {tag, a_size})
        if size + 1 == limit then
          break
        end -- TODO this is not pretty
      end      
    else
      break
    end
    if size + 1 == limit then -- this is not pretty at all
      break
    end
  end 
  parsed_data.size = function () return size end
  parsed_data.tags = tags
  print("Dataset loaded.")
  f:close()
  return parsed_data
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