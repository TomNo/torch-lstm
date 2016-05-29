require 'torch'
require 'nn'
require 'Step'
require 'AddLinear'

--[[
-- Class GruStep executes all operations that need to be done in every timestep,
-- for Gru.
-- Reset operation is perfomed first, similarly to http://arxiv.org/abs/1412.3555
--]]

local GruStep = torch.class('nn.GruStep', 'nn.Step')

function GruStep:__init(layerSize)
    nn.Step.__init(self, layerSize)
    self.inputSize = 3 * layerSize

    self.rzActs = nn.AddLinear(layerSize, 2 * layerSize)
    self.rzSigmoid = nn.Sigmoid()
    self.aTanh = nn.Tanh()
    self.aActs = nn.AddLinear(layerSize, layerSize)

    self.rScale = nn.CMulTable()
    self.zScale = nn.CMulTable()
    self.zpScale = nn.CMulTable()
    self.sum = nn.CAddTable(true)

    self.rzInt = {{}, {1, 2*layerSize} }
    self.rInt = {{}, {1, layerSize} }
    self.zInt = {{}, {layerSize + 1, 2 *layerSize} }
    self.aInt = {{}, {layerSize * 2 + 1, layerSize * 3}}

    local mNames = {"rzActs", "rzSigmoid", "aTanh", "aActs", "rScale",
        "zpScale", "sum" }
    for i=1, #mNames do
        self:add(self[mNames[i]])
    end
    self.gradInput = torch.Tensor()
end


function GruStep:updateOutput(input)
    local aInput = self:currentInput(input)
    self.input = aInput[1]
    self.pOutput = aInput[2]
    self.rzActs:forward({self.input[self.rzInt], self.pOutput})
    self.rzSigmoid:forward(self.rzActs.output)
    self.rScale:forward({self.rzSigmoid.output[self.rInt], self.pOutput})
    self.aActs:forward({self.input[self.aInt], self.rScale.output})
    self.aTanh:forward(self.input[self.aInt])
    self.zScale:forward({self.pOutput, self.rzSigmoid.output[self.zInt]})
    self.rzSigmoid.output[self.zInt]:mul(-1)
    self.rzSigmoid.output[self.zInt]:add(1)
    self.zpScale:forward({self.aTanh.output, self.rzSigmoid.output[self.zInt]})
    self.sum:forward({self.zScale.output, self.zpScale.output})
    self.output = self.sum.output
    return self.sum.output
end

function GruStep:backward(input, gradOutput, scale)
    local function backward(module, input, gradOutput)
        module:backward(input, gradOutput, scale)
    end
    if self.nStep then
        local deltas = self.nStep():getOutputDeltas()
        local rInt = {{1, math.min(deltas:size(1), gradOutput:size(1))}}
        gradOutput[rInt]:add(deltas[rInt])
    end
    self.gradInput:resizeAs(input)
    backward(self.sum, {self.zScale.output, self.zpScale.output}, gradOutput)
    backward(self.zpScale,
             {self.aTanh.output, self.rzSigmoid.output[self.zInt]},
              self.sum.gradInput[2])
    self.zpScale.gradInput[2]:mul(-1)
    self.rzSigmoid.output[self.zInt]:add(-1)
    self.rzSigmoid.output[self.zInt]:mul(-1)
    backward(self.zScale,
             {self.pOutput, self.rzSigmoid.output[self.zInt]},
             self.sum.gradInput[1])
    self.aTanh.gradInput = self.gradInput[self.aInt]
    backward(self.aTanh, self.input[self.aInt], self.zpScale.gradInput[1])
    backward(self.aActs, {self.input[self.aInt], self.rScale.output},
             self.aTanh.gradInput)
    backward(self.rScale, {self.rzSigmoid.output[self.rInt], self.pOutput},
             self.aActs.gradInput[2])
    self.rzSigmoid.gradInput = self.gradInput[self.rzInt]
    self.zpScale.gradInput[2]:add(self.zScale.gradInput[2])
    backward(self.rzSigmoid , self.rzActs.output,
             self.rScale.gradInput[1]:cat(self.zpScale.gradInput[2]))
    backward(self.rzActs, {self.input[self.rzInt], self.pOutput},
             self.rzSigmoid.gradInput)
    self.rzActs.gradInput[2]:add(self.rScale.gradInput[2])
    self.rzActs.gradInput[2]:add(self.zScale.gradInput[1])
    return self.gradInput
end

function GruStep:getGradInput()
    return self.gradInput
end

function GruStep:getOutputDeltas()
    return self.rzActs.gradInput[2]
end


--eof
