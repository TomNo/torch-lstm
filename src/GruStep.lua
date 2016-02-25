require 'torch'
require 'nn'


-- TODO maybe gradOutput should be negated
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


local GruStep = torch.class('nn.GruStep', 'nn.Sequential')

function GruStep:__init(layerSize)
    nn.Sequential.__init(self)
    self.layerSize = layerSize
    self.inputSize = 3 * layerSize
    self.zTensor = torch.zeros(1)
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
    gates:add(nn.CAddTable())
    gates:add(nn.Sigmoid())
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
    local nonLinearity = nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()):add(nn.Tanh())
    self:add(nn.ConcatTable():add(nonLinearity):add(nn.SelectTable(3)):add(nn.SelectTable(4)))
    self:add(nn.ConcatTable():add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CMulTable())):add(nn.SelectTable(3)))
    self:add(nn.CAddTable())
end


function GruStep:updateOutput(input)
    nn.Sequential.updateOutput(self, self:currentInput(input))
    return self.output
end


function GruStep:updateGradInput(input, gradOutput)
    if self.nStep then
        local nGradOutput = self.nStep():getOutputDeltas()
        gradOutput:add(nGradOutput)
    end
    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i = #self.modules - 1, 1, -1 do
        local previousModule = self.modules[i]
        currentGradOutput = currentModule:updateGradInput(previousModule.output,
            currentGradOutput)
        currentModule.gradInput = currentGradOutput
        currentModule = previousModule
    end
    currentGradOutput = currentModule:updateGradInput(self:currentInput(input),
        currentGradOutput)
    self.gradInput = currentGradOutput
    return currentGradOutput
end


function GruStep:accGradParameters(input, gradOutput, scale)
    nn.Sequential.accGradParameters(self, self:currentInput(input), gradOutput,
        scale)
end


function GruStep:backward(input, gradOutput, scale)
    scale = scale or 1
    self:updateGradInput(input, gradOutput)
    self:accGradParameters(input, gradOutput, scale)
    return self.gradInput
end


function GruStep:getGradInput()
    return self.gradInput[1]
end


function GruStep:getOutputDeltas()
    return self.gradInput[2]
end


function GruStep:currentInput(input)
    local pOutput
    if self.pStep then
        local pStep = self.pStep()
        pOutput = pStep.output
    else
        pOutput = self.zTensor:repeatTensor(input:size(1), self.layerSize)
    end
    return {input, pOutput}
end

--eof
