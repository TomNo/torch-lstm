require 'torch'
require 'nn'
require 'LinearScale'


local Bias = torch.class("nn.Bias", "nn.LinearScale")


function Bias:updateOutput(input)
    if input:dim() == 1 then
        error('Input must be matrix')
    elseif input:dim() == 2 then
        input:add(self.weight:repeatTensor(input:size(1)))
    else
        error('Input must be vector or matrix')
    end
    self.output = input
    return self.output
end


function Bias:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput = gradOutput
    end
    return self.gradInput
end


function Bias:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    self.gradWeight:add(scale, gradOutput:sum(1))
end


--eof
