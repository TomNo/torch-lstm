require 'torch'
require 'nn'
require 'BatchRecurrent'
require 'SharedInput'


--[[
-- Bidirectional class, represents bidirectional recurrent neural networks.
-- One module always propagates its input in the opposite direction.
 ]]


local Bidirectional = torch.class("nn.Bidirectional", "nn.BatchRecurrent")


function Bidirectional:__init(inputSize, layerSize, hist, bNorm, dropout)
    assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
    nn.Sequential.__init(self)
    self.aSize = layerSize / 2
    self.dropout = dropout or 0
    self.bNorm = bNorm or false
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history = hist
    self:_setActualModule()
    self.fModule = nn.Sequential()
    self.fModule:add(nn.Linear(inputSize, self.aModule.inputSize, false))
    if bNorm then
        self.fModule:add(nn.BatchNormalization(self.aModule.inputSize))
    end
     if self.dropout ~= 0 then
        self.fModule:add(nn.Dropout(self.dropout))
     end
    self.fModule:add(self.aModule)

    self.bModule = nn.Sequential()
    self.bModule:add(nn.Linear(inputSize, self.aModule.inputSize, false))
    if bNorm then
        self.bModule:add(nn.BatchNormalization(self.aModule.inputSize))
    end
    if self.dropout ~= 0 then
        self.bModule:add(nn.Dropout(self.dropout))
     end
    local bModule = self.aModule:clone()
    bModule.revert = true
    self.bModule:add(bModule)

    local concat = nn.SharedInput()
    concat:add(self.fModule)
    concat:add(self.bModule)
    self:add(concat)
    self:add(nn.JoinTable(2, 2))
end


--eof
