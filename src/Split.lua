require 'torch'
require 'nn'

local Split = torch.class('nn.Split', 'nn.Module')

function Split:updateOutput(input)
    local size = input:size(2)
    self.output = {
        input[{ {}, { 1, size / 2 } }],
        input[{ {}, { size / 2 + 1, size } }]
    }
    return self.output
end

function Split:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    local size = input:size(2)
    self.gradInput[{ {}, { 1, size / 2} }]:copy(gradOutput[1])
    self.gradInput[{ {}, { size / 2 + 1, size } }]:copy(gradOutput[2])
    return self.gradInput
end


--eof
