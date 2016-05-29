require 'torch'
require 'nn'


--[[
-- AddLinear class modifies standard Linear class,
-- so the output of affine transformation can be added directly to the
-- prepared Tensor, which is supplied during forward pass.
--
-- During forward pass is expected table with two Tensors,
-- first item should be the Tensor into which the result should be added
-- and second should be input to the affine transformation.
--
-- GradInput consist of table, which contains two Tensors representing gradInput
-- for the input modules.
 ]]

local AddLinear = torch.class("nn.AddLinear", "nn.Linear")


function AddLinear:__init(inputSize, outputSize, bias)
    nn.Linear.__init(self, inputSize, outputSize, bias)
    self.gradInputBackup = self.gradInput
end


--copy/paste as this function is not accessible within module
local function updateAddBuffer(self, input)
    local nframe = input:size(1)
    self.addBuffer = self.addBuffer or input.new()
    if self.addBuffer:nElement() ~= nframe then
        self.addBuffer:resize(nframe):fill(1)
    end
end


function AddLinear:updateOutput(input)
    self.output = input[1]
    input = input[2]
    if input:dim() == 1 then
        if self.bias then self.output:add(self.bias) end
        self.output:addmv(1, self.weight, input)
    elseif input:dim() == 2 then
        updateAddBuffer(self, input)
        self.output:addmm(self.output, 1, input, self.weight:t())
        --addr??
        if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
    else
        error('input must be vector or matrix')
    end

    return self.output
end


function AddLinear:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInputBackup
    nn.Linear.updateGradInput(self, input[2], gradOutput)
    self.gradInput = { gradOutput, self.gradInput }
    return self.gradInput
end


function AddLinear:accGradParameters(input, gradOutput, scale)
    input = input[2]
    nn.Linear.accGradParameters(self, input, gradOutput, scale)
end

--eof
