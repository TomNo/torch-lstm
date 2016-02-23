require 'torch'
require 'nn'


local GruStep = torch.class('nn.GruStep', 'nn.Sequential')

function GruStep:__init(layerSize)
    nn.Sequential.__init(self)
    self.layerSize = layerSize
    self.zTensor = torch.zeros(1)
    -- hidden to hidden activations
    local hActs = nn.Sequential():add(nn.Linear(layerSize, 2 * layerSize))
    -- split input activation to two parts - one is going to be used for gates
    -- the other one for input tanh activation
    local sInput = nn.

end


function GruStep:updateOutput(input)
    nn.Sequential.updateOutput(self, self:currentInput(input))
    return self.output
end


function GruStep:updateGradInput(input, gradOutput)
    local nGradOutput, nCellGradOutput
    if self.nStep then
        nGradOutput = self.nStep():getOutputDeltas()
    else
        nGradOutput = self.zTensor:repeatTensor(input:size(1), self.layerSize)
    end
    gradOutput:add(nGradOutput)
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
    return self.gradInput[1][1]
end


function GruStep:getOutputDeltas()
    return self.gradInput[1][2]
end


function GruStep:currentInput(input)
    local pOutput
    if self.pStep then
        local pStep = self.pStep()
        pOutput = pStep.output
    else
        local zInput = self.zTensor:repeatTensor(input:size(1), self.layerSize)
        pOutput = zInput
    end
    return { input, pOutput }
end

--eof
