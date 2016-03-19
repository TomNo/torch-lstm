require 'torch'
require 'nn'
require 'Step'
require 'Bias'


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
    local inputActs = nn.Sequential():add(nn.Reshape(3, layerSize)):add(nn.SplitTable(1,2))
    local inputs = nn.ParallelTable():add(inputActs):add(nn.Identity())
    self:add(inputs)
    self:add(nn.FlattenTable())
    self:add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.NarrowTable(2,3)))
    -- hidden to hidden activations
    -- set bias to 1 because of the forget(reset) gate activation
    local hActs = nn.Linear(layerSize, 2 * layerSize)
    hActs.bias:fill(1)
    local gates = nn.Sequential()
    local gInputs = nn.ConcatTable()
    gInputs:add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.JoinTable(2)))
    gInputs:add(nn.Sequential():add(nn.SelectTable(3)):add(hActs))
    gates:add(gInputs)
    gates:add(nn.CAddTable(true))
    gates:add(nn.Sigmoid(true))
    gates:add(nn.Split())
    -- now we have gate activations - time to apply them
    local gApp = nn.Sequential():add(nn.ConcatTable():add(gates):add(nn.SelectTable(3)))
    gApp:add(nn.FlattenTable())
    local updateGateTransform = nn.Sequential():add(nn.SelectTable(1)):add(nn.UpdateGateTransform())
    local forgetGate = nn.Sequential():add(nn.NarrowTable(2, 2)):add(nn.CMulTable()):add(nn.Linear(layerSize, layerSize, false))
    local updateGate = nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(3))):add(nn.CMulTable())
    local concat = nn.ConcatTable()
    concat:add(forgetGate)
    concat:add(updateGateTransform)
    concat:add(updateGate)
    gApp:add(concat)
    self:add(nn.ParallelTable():add(nn.Identity()):add(gApp))
    self:add(nn.FlattenTable())
    local nonLinearity = nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable(true)):add(nn.Bias(layerSize)):add(nn.Tanh(true))
    self:add(nn.ConcatTable():add(nonLinearity):add(nn.SelectTable(3)):add(nn.SelectTable(4)))
    self:add(nn.ConcatTable():add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CMulTable())):add(nn.SelectTable(3)))
    self:add(nn.CAddTable(true))
end


--eof
