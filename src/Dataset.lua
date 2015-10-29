require 'torch'
require 'hdf5'


local Dataset = torch.class('Dataset')

function Dataset:__init(filename)
  self.data = hdf5.open(filename):all()
end

function Dataset:get()
  if self.parsed_data == nil then
    self.parsed_data = {}
    local size = 0 
    for _, item in pairs(self.data) do
      local cols = item.cols[1]
      local rows = item.rows[1]
      local data = item.data:reshape(rows, cols)
      for i=1, rows do
        size = size + 1
        table.insert(self.parsed_data, {[1]=data[i], [2]=item.labels[{{i}}]})
      end
    end
    self.parsed_data.size = function () return size end
  end
  return self.parsed_data
end