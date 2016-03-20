require 'torch'
require 'nn'


local Step = torch.class('nn.Step', 'nn.Sequential')


function Step:__init(layerSize)
    self.layerSize = layerSize
    self.zTensor = torch.zeros(1)
    nn.Sequential.__init(self)
end


function Step:updateOutput(input)
    nn.Sequential.updateOutput(self, self:currentInput(input))
    return self.output
end


function Step:updateGradInput(input, gradOutput)
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


function Step:accGradParameters(input, gradOutput, scale)
    nn.Sequential.accGradParameters(self, self:currentInput(input), gradOutput,
        scale)
end


function Step:backward(input, gradOutput, scale)
   scale = scale or 1
   if self.nStep then
        local nGradOutput = self.nStep():getOutputDeltas()
        gradOutput:add(nGradOutput)
   end
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(self:currentInput(input), currentGradOutput, scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end


function Step:getGradInput()
    return self.gradInput[1]
end


function Step:getOutputDeltas()
    return self.gradInput[2]
end


function Step:currentInput(input)
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
