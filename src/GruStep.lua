require 'torch'
require 'nn'
require 'Step'
require 'Bias'
require 'Split'
require 'AddLinear'

local UpdateGateTransform = torch.class("nn.UpdateGateTransform", "nn.Identity")


-- (1 - z)
function UpdateGateTransform:updateOutput(input)
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:mul(-1)
    self.output:add(1)
    return self.output
end


function UpdateGateTransform:updateGradInput(input, gradOutput)
    nn.Identity.updateGradInput(self, input, gradOutput)
    self.gradInput:mul(-1)
    return self.gradInput
end


local GruStep = torch.class('nn.GruStep', 'nn.Step')


function GruStep:__init(layerSize)
    nn.Step.__init(self, layerSize)
    self.inputSize = 3 * layerSize
    local zGate = nn.Sequential()
    zGate:add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(4)):add(nn.SelectTable(4)))
    zGate:add(nn.ConcatTable():add(nn.NarrowTable(1,2)):add(nn.SelectTable(3)))
    zGate:add(nn.ParallelTable():add(nn.Sequential():add(nn.AddLinear(layerSize, layerSize)):add(nn.Sigmoid(true))):add(nn.Identity()))
    zGate:add(nn.ConcatTable():add(nn.NarrowTable(1,2)):add(nn.SelectTable(1)))
    zGate:add(nn.ParallelTable():add(nn.CMulTable()):add(nn.UpdateGateTransform()))

    local rHidden = nn.AddLinear(layerSize, layerSize)
    rHidden.bias:fill(1)
    local rGate =nn.Sequential():add(nn.ConcatTable():add(nn.NarrowTable(1,2)):add(nn.SelectTable(3)))
    rGate:add(nn.ParallelTable():add(nn.Sequential():add(rHidden):add(nn.Sigmoid(true))):add(nn.Identity()))
    rGate:add(nn.CMulTable())

    local h = nn.Sequential()
    h:add(nn.ConcatTable():add(nn.SelectTable(3)):add(nn.SelectTable(2)):add(nn.SelectTable(4)):add(nn.SelectTable(4)))
    h:add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.NarrowTable(2,3)))
    h:add(nn.ParallelTable():add(nn.Identity()):add(rGate))
    h:add(nn.AddLinear(layerSize, layerSize))
    h:add(nn.Tanh())

    self:add(nn.ConcatTable():add(zGate):add(h))
    self:add(nn.FlattenTable())
    self:add(nn.ConcatTable():add(nn.NarrowTable(2,2)):add(nn.SelectTable(1)))
    self:add(nn.ParallelTable():add(nn.CMulTable()):add(nn.Identity()))
    self:add(nn.FlattenTable())
    self:add(nn.CAddTable(true))
end


function GruStep:getGradInput()
    return {self.gradInput[1], self.gradInput[2], self.gradInput[3]}
end


function GruStep:getOutputDeltas()
    return self.gradInput[4]
end


--eof
