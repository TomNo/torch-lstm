require 'torch'
require 'nn'
require 'Step'
require 'Steps'
require 'BatchRecurrent'
require 'Bidirectional'


local RecSteps = torch.class("nn.RecSteps", "nn.Steps")


function RecSteps:__init(aType, layerSize, history)
    self.aType = aType
    nn.Steps.__init(self, layerSize, history)
end


function RecSteps:_setStepModule()
    self.step = nn.RecStep(self.aType, self.layerSize)
end


local RecStep = torch.class("nn.RecStep", "nn.Step")


function RecStep:__init(aType, layerSize)
    nn.Step.__init(self, layerSize)
    self.inputSize = layerSize
    self:add(nn.ParallelTable():add(nn.Identity()):add(nn.Linear(layerSize, layerSize)))
    self:add(nn.CAddTable())
    self:add(aType())
end


local RecLayer = torch.class("nn.RecLayer", "nn.BatchRecurrent")


function RecLayer:__init(actType, inputSize, layerSize, hist, bNorm)
    self.aType = actType
    nn.BatchRecurrent.__init(self, inputSize, layerSize, hist, bNorm)
end


function RecLayer:_setActualModule()
    self.aModule = nn.RecSteps(self.aType, self.layerSize, self.history)
end


local BiRecLayer = torch.class("nn.BiRecLayer", "nn.Bidirectional")

function BiRecLayer:__init(aType, inputSize, layerSize, hist, bNorm)
    self.aType = aType
    nn.Bidirectional.__init(self, inputSize, layerSize, hist, bNorm)
end


function BiRecLayer:_setActualModule()
    self.aModule = nn.RecSteps(self.aType, self.aSize, self.history)
end

--eof
