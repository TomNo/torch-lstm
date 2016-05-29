require 'hdf5'
require 'torch'


--[[
-- OutputDs class is used for saving the forward pass in to the file.
 ]]

local OutputDs = torch.class("OutputDs")


function OutputDs:__init(filename)
    self.filename = filename
    self.file = hdf5.open(self.filename, "w")
end


function OutputDs:save(data, tag)
    local rows = torch.Tensor(1)
    rows[1] = data:size(1)
    local cols = torch.Tensor(1)
    cols[1] = data:size(2)
    self.file:write(tag .. "/data", data)
    self.file:write(tag .. "/rows", rows)
    self.file:write(tag .. "/cols", cols)
end


function OutputDs:close()
    self.file:close()
end


--eof
