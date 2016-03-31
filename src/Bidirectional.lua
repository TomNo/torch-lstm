require 'torch'
require 'nn'
require 'Split'
require 'Revert'
require 'BatchRecurrent'
require 'SharedInput'


local Bidirectional = torch.class("nn.Bidirectional", "nn.Sequential")


function Bidirectional:__init(inputSize, layerSize, hist, bNorm)
    assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
    nn.Sequential.__init(self)
    self.aSize = layerSize / 2
    self.bNorm = bNorm or false
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history = hist
    self:_setActualModule()
    self.bModule = self.aModule:clone()
    self.bModule.revert = true
    local concat = nn.SharedInput()
    concat:add(self.aModule)
    concat:add(self.bModule)
    self:add(concat)
    self:add(nn.JoinTable(2, 2))
end


function Bidirectional:_setActualModule()
    error("This method muset be overiden in superclass.")
end


function Bidirectional:__tostring__()
    return torch.type(self) .. string.format('(%d -> %d, BatchNormalized=%s)',
                                             self.inputSize,
                                             self.layerSize,
                                             self.bNorm)
end

--eof
