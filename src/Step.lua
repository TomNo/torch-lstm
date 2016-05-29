require 'torch'
require 'nn'


local Step = torch.class('nn.Step', 'nn.Sequential')


function Step:__init(layerSize)
    self.layerSize = layerSize
    self.zTensor = torch.Tensor()
    nn.Sequential.__init(self)
end


function Step:updateOutput(input)
    nn.Sequential.updateOutput(self, self:currentInput(input))
    return self.output
end


function Step:updateGradInput(input, gradOutput)
    error("Method is not supported, use backward instead.")
end


function Step:accGradParameters(input, gradOutput, scale)
    error("Method is not supported, use backward instead.")
end


function Step:backward(input, gradOutput, scale)
   scale = scale or 1
   if self.nStep then
        local deltas = self.nStep():getOutputDeltas()
        local rInt = {{1, math.min(deltas:size(1), gradOutput:size(1))}}
        gradOutput[rInt]:add(deltas[rInt])
   end
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:backward(previousModule.output,
          currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(self:currentInput(input),
       currentGradOutput[{{1, input:size(1)}}], scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end


function Step:getGradInput()
    return self.gradInput[1]
end


function Step:getOutputDeltas()
    return self.gradInput[2]
end


function Step:adaptZTensor()
    if self.zTensor:dim() == 0 then
        self.zTensor:resize(1, self.layerSize)
        self.zTensor:zero()
    end
end

function Step:currentInput(input)
    local pOutput
    if self.pStep then
        local pStep = self.pStep()
        pOutput = pStep.output
    else
        self:adaptZTensor()
        pOutput = self.zTensor:expand(input:size(1), self.layerSize)
    end
    local rInt = {{1, input:size(1)} }
    if pOutput:size(1) < input:size(1) then
        local pSize = pOutput:size(1)
        pOutput:resize(input:size(1), pOutput:size(2))
        pOutput[{{pSize + 1, input:size(1)}}]:zero()
    end

    return {input, pOutput[rInt]}
end


--eof
