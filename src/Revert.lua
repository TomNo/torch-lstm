require 'torch'
require 'nn'


local Revert = torch.class('nn.Revert', 'nn.Module')


function Revert:__init()
  nn.Module.__init(self)
end


function Revert:revertTensor(input, output)
  output:resizeAs(input)
  for i=1,input:size(1) do
    output[i] = input[input:size(1) + 1 - i]
  end
  return output
end


function Revert:updateGradInput(input, gradOutput)
  self:revertTensor(gradOutput, self.gradInput)
  return self.gradInput
end


function Revert:updateOutput(input)
  self:revertTensor(input, self.output)
  return self.output
end