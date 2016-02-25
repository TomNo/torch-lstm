require 'torch'
require 'nn'
require 'Split'
require 'Revert'
require 'BatchRecurrent'


local Bidirectional = torch.class("nn.Bidirectional", "nn.BatchRecurrent")


function Bidirectional:__init(inputSize, layerSize, hist, bNorm)
    assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
    nn.Sequential.__init(self)
    self.aSize = layerSize / 2
    self.bNorm = bNorm or false
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history = hist
    self:_setActualModule()
    self:add(nn.Linear(inputSize, self.aModule.inputSize * 2, false))
    if bNorm then
        self:add(nn.BatchNormalization(self.aModule.inputSize *2))
    end
    self:add(nn.Split())
    self.fModule = self.aModule
    self.bModule = nn.Sequential()
    self.bModule:add(nn.Revert())
    self.bModule:add(self.aModule:clone())
    self.bModule:add(nn.Revert())
    local pTable = nn.ParallelTable()
    pTable:add(self.fModule)
    pTable:add(self.bModule)
    self:add(pTable)
    self:add(nn.JoinTable(2, 2))
end


--eof
