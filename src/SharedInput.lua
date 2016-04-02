require 'nn'
require 'torch'

-- similair to concat table but simpler and memory eficient backward
local SharedInput = torch.class('nn.SharedInput', 'nn.Container')


function SharedInput:updateOutput(input)
    self.output = {}
    for i=1, #self.modules do
        table.insert(self.output, self.modules[i]:forward(input))
    end
    return self.output
end


function SharedInput:updateGradInput(...)
    error("This method should never be called, user backward instead.")
end


function SharedInput:accGradParameters(...)
    error("This method should never be called, user backward instead.")
end


function SharedInput:updateOutput(input)
    self.output = {}
    for i=1, #self.modules do
        table.insert(self.output, self.modules[i]:forward(input))
    end
    return self.output
end


function SharedInput:backward(input, gradOutput, scale)
    self.gradInput:resizeAs(input)
    self.modules[1]:backward(input, gradOutput[1], scale)
    self.gradInput:copy(self.modules[1].gradInput)
    for i=2, #self.modules do
        self.modules[i]:backward(input, gradOutput[i], scale)
        self.gradInput:add(self.modules[i].gradInput)
    end
    return self.gradInput
end


--eof
