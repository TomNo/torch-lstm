require 'torch'
require 'nn'


local MaskedCECriterion, parent = torch.class("nn.MaskedCECriterion", "nn.CrossEntropyCriterion")


-- CrossEntropy does not acept '0' as label -> gradients are somewhat stacked
-- in the gpu version and it does not work on cpu at all -> create mask and
-- initialy set every label to 1 instead of 0 -> this will not affect the error

function MaskedCECriterion:__init(weights)
    parent.__init(self, weights)
    self.mask = torch.Tensor()
    self.nll.sizeAverage = false -- size averaging must be done manually
end

function MaskedCECriterion:updateOutput(input, target)
    input = input:squeeze()
    if torch.type(target) == 'number' then
        error("Only batch mode is supported.")
    end
    target = target:squeeze()
    self.mask:resizeAs(target)
    self.mask:copy(target)
    self.mask = self.mask:view(input:size(1), 1)
    self.mask:clamp(0,1)
    self.lsm:updateOutput(input)
    self.lsm.output:cmul(self.mask:expand(input:size(1), input:size(2)))
    -- remove 0 labels
    target:clamp(1, 1e10)
    self.nll:updateOutput(self.lsm.output, target)
    self.nll.output = self.nll.output / self.mask:sum()
    self.output = self.nll.output
    return self.output
end


function MaskedCECriterion:updateGradInput(input, target)
    local size = input:size()
    input = input:squeeze()
    target = type(target) == 'number' and target or target:squeeze()
    self.nll:updateGradInput(self.lsm.output, target)
    self.nll.gradInput:cmul(self.mask:expand(self.nll.gradInput:size(1), self.nll.gradInput:size(2)))
    self.lsm:updateGradInput(input, self.nll.gradInput)
    self.gradInput:view(self.lsm.gradInput, size)
    self.gradInput:mul(1/self.mask:sum())
    return self.gradInput
end