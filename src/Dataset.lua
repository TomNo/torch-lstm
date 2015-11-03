require 'torch'
require 'hdf5'


local Dataset = torch.class('Dataset')

function Dataset:__init(filename)
  self.filename = filename
end

-- reads the data from input file and returns the dataset
function Dataset:get(limit)
  local f = hdf5.open(self.filename)
  local data = f:all()
  f:close()
  local parsed_data = {}
  local tags = {}
  local size = 0 
  for tag, item in pairs(data) do
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
      if size == limit then
        a_size = i
        break
      end
    end
    table.insert(tags, {tag, a_size})
    if size == limit then
      break
    end -- TODO this is not pretty
  end
  parsed_data.size = function () return size end
  parsed_data.tags = tags
  return parsed_data
end

-- saves the data into output file specified on initialization
function Dataset:save(data, tags)
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
end