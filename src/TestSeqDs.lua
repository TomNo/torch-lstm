require 'hdf5'
require 'torch'


torch.setdefaulttensortype('torch.FloatTensor')

-- Module for forward pass dataset


local TestSeqDs = torch.class('TestSeqDs')


function TestSeqDs:__init(filename)
    self.filename = filename
    self.iterator = 0
    self.file = hdf5.open(self.filename)
end


function TestSeqDs:reset()
    self.iterator = 0
end


function TestSeqDs:getSeq()
    local iIndex = "/" .. tostring(self.iterator)
    local status, data = pcall(self.file.read, self.file, iIndex)
    self.iterator = self.iterator + 1
    if status then
        local tmp = data:all()
        -- we need to move tag to the special field
        -- this is not possible because of problematic implementation of hdf5
        -- in lua - no string allowed as parameters
        for tag, item in pairs(tmp) do
            item.tag = tag
            item.data = item.data:resize(item.rows[1], item.cols[1])
            return item
        end
    else
        return nil
    end
end


--eof
