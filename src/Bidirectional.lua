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
    self.fModule = nn.Sequential()
    self.fModule:add(nn.Linear(inputSize, self.aModule.inputSize, false))
    if bNorm then
        self.fModule:add(nn.BatchNormalization(self.aModule.inputSize))
    end
    self.fModule:add(self.aModule)

    self.bModule = nn.Sequential()
    self.bModule:add(nn.Linear(inputSize, self.aModule.inputSize, false))
    if bNorm then
        self.bModule:add(nn.BatchNormalization(self.aModule.inputSize))
    end
    self.bModule:add(nn.Revert())
    self.bModule:add(self.aModule:clone())
    self.bModule:add(nn.Revert())

    local concat = nn.ConcatTable()
    concat:add(self.fModule)
    concat:add(self.bModule)
    self:add(concat)
    self:add(nn.JoinTable(2, 2))
end


--eof
