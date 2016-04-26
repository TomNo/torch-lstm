require 'torch'
require 'nn'
require 'warp_ctc'
require 'utils'

--TODO optimize validation vs training
CtcCriterion, parent = torch.class("nn.CtcCriterion","nn.Criterion")


function CtcCriterion:__init(history)
    parent.__init(self)
    self.history = history
    self.grads = torch.Tensor()
    self.aInput = torch.Tensor()
    self.ctc = cpu_ctc
end

function CtcCriterion:forward(input, target, sizes, bSizes)
    self.sizes = sizes
    self.bSizes = bSizes
    return self:updateOutput(input, target, sizes)
end

function CtcCriterion:updateOutput(input, target)
    self:_unfoldInput(input)
    self.grads:resizeAs(self.aInput)
    local costs = self.ctc(self.aInput, self.grads, target, self.sizes)
    -- remove nans and infs as ctc is sometimes pretty unstable
    costs = utils.removeNonNumbers(costs)
    self.output = utils.sumTable(costs)
    self.output = self.output / #costs
    return self.output
end

function CtcCriterion:_unfoldInput(input)
    self.aInput:resize(#self.bSizes * #self.sizes, input:size(2))
    self.aInput:zero()
    local bIndex = 0
    local aIndex = 0
    for i=1, #self.bSizes do
        local aInt = {{aIndex + 1, aIndex + self.bSizes[i]}}
        local bInt = {{bIndex + 1, bIndex + self.bSizes[i]}}
        self.aInput[aInt]:copy(input[bInt])
        bIndex = bIndex + self.bSizes[i]
        aIndex = aIndex + #self.sizes
    end
    return self.aInput
end

function CtcCriterion:_foldGradInput(input, grads)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    local aIndex = 0
    local bIndex = 0
    for i=1, #self.bSizes do
        local aInt = {{aIndex + 1, aIndex + self.bSizes[i]}}
        local bInt = {{bIndex + 1, bIndex + self.bSizes[i]} }
        self.gradInput[bInt]:copy(grads[aInt])
        bIndex = bIndex + self.bSizes[i]
        aIndex = aIndex + #self.sizes
    end
    return self.gradInput
end

function CtcCriterion:updateGradInput(input, target)
   return self:_foldGradInput(input, self.grads)
end

function CtcCriterion:cuda()
    local toCuda = parent.cuda(self)
    toCuda.ctc = gpu_ctc
    return toCuda
end

function CtcCriterion:float()
    local toFloat = parent.float(self)
    toFloat.ctc = cpu_ctc
    return toFloat
end

function CtcCriterion:double()
    local toDouble = parent.double(self)
    toDouble.ctc = cpu_ctc
    return toDouble
end

-- extract ctc labels from the regular labels that has for every frame some
-- label
--function CtcCriterion:_getLabels(target)
--    local labels = {}
--    local seQCount = target:size(1) / self.history
--    -- initialize and insert first label
--    for i=1, seQCount do
--        table.insert(labels, {target[i]})
--    end
--
--    -- iterate over labels and if target is not the same as in the previous
--    -- timestep insert it
--    for i=1, self.history - 1 do
--        for y=1, seQCount do
--            if target[i * seQCount + y] ~= target[(i-1) * seQCount + y] then
--                table.insert(labels[y], target[i * seQCount + y])
--            end
--        end
--    end
--    return labels
--end

--eof
