require 'torch'
require 'nn'

local Split = torch.class('nn.Split', 'nn.Module')


function Split:__init(count)
    self.count = count or 2
    nn.Module.__init(self)
end


function Split:updateOutput(input)
    local size = input:size(2)
    self.output = {}
    for i=1, self.count do
        local sInt =  size * ((i - 1) / self.count) + 1
        local eInt = size * (i / self.count)
        self.output[i] = input[{{},{sInt,eInt}}]
    end

    return self.output
end

function Split:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    local size = input:size(2)
    for i=1, self.count do
        local sInt =  size * ((i - 1) / self.count) + 1
        local eInt = size * (i / self.count)
        self.gradInput[{{},{sInt, eInt}}]:copy(gradOutput[i])
    end
    return self.gradInput
end


--eof
