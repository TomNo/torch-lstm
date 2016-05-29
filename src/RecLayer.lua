require 'torch'
require 'nn'
require 'Step'
require 'Steps'
require 'BatchRecurrent'
require 'Bidirectional'
require 'AddLinear'


--[[
-- This module represent classic rnn and
-- its identity variant irnn.
 ]]

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
    self:add(nn.AddLinear(layerSize, layerSize))
    self:add(aType())
end


local RecLayer = torch.class("nn.RecLayer", "nn.BatchRecurrent")


function RecLayer:__init(actType, inputSize, layerSize, hist, bNorm, dropout)
    self.aType = actType
    nn.BatchRecurrent.__init(self, inputSize, layerSize, hist, bNorm, dropout)
end


function RecLayer:_setActualModule()
    self.aModule = nn.RecSteps(self.aType, self.layerSize, self.history)
end


local BiRecLayer = torch.class("nn.BiRecLayer", "nn.Bidirectional")

function BiRecLayer:__init(aType, inputSize, layerSize, hist, bNorm, dropout)
    self.aType = aType
    nn.Bidirectional.__init(self, inputSize, layerSize, hist, bNorm, dropout)
end


function BiRecLayer:_setActualModule()
    self.aModule = nn.RecSteps(self.aType, self.aSize, self.history)
end


--http://arxiv.org/pdf/1504.00941.pdf
local IRecLayer = torch.class("nn.IRecLayer", "nn.RecLayer")


function ireset(self)
    self.bias:zero()
    self.weight:copy(torch.eye(self.weight:size(1)))
end


function IRecLayer:__init(actType, inputSize, layerSize, hist, bNorm)
    nn.RecLayer.__init(self, actType, inputSize, layerSize, hist, bNorm)
    self:findModules("nn.AddLinear")[1].reset = ireset
end


local BIRecLayer = torch.class("nn.BIRecLayer", "nn.BiRecLayer")

function BIRecLayer:_setActualModule()
    self.aModule = nn.RecSteps(self.aType, self.aSize, self.history)
    self.aModule:findModules("nn.AddLinear")[1].reset = ireset
end

--eof
