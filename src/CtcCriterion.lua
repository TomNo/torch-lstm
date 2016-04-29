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


function CtcCriterion:forwardOnly(...)
    self.fOnly = true
    self:forward(...)
    self.fOnly = false
    return self.output
end

function CtcCriterion:updateOutput(input, target)
    self:_unfoldInput(input)
    local costs
    if not self.fOnly then
        self.grads:resizeAs(self.aInput)
        costs = self.ctc(self.aInput, self.grads, target, self.sizes)
        if self.grads:ne(self.grads):sum() > 0 then
            print(self.aInput)
            error("Nans occured in self.grads.")
        end
    else
        costs = self.ctc(self.aInput, self.aInput.new(), target, self.sizes)
    end
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


--eof
