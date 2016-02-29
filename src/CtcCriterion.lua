require 'torch'
require 'nn'
require 'warp_ctc'

--TODO optimize validation vs training
CtcCriterion, parent = torch.class("nn.CtcCriterion","nn.Criterion")


function sumTable(tb)
    local acc
    for k, v in ipairs(tb) do
        if 1 == k then
            acc = v
        else
            acc = acc + v
        end
    end
    return acc
end


function CtcCriterion:__init(history)
    parent.__init(self)
    self.history = history
    self.grads = torch.Tensor()
    self.ctc = cpu_ctc
    -- remove the log operation of the softmax
    self.exp = nn.Exp()

end

--(acts, grads, labels, sizes)
function CtcCriterion:updateOutput(input, target)
    self.grads:resizeAs(input)
    local sizes = {}
    local labels = self:_getLabels(target)
    local seqCount = input:size(1) / self.history
    for _=1, seqCount do
        table.insert(sizes, self.history)
    end
    local costs = self.ctc(self.exp(input), self.grads, labels, sizes)
    self.output = sumTable(costs)
    return self.output
end


-- extract ctc labels from the regular labels that has for every frame some
-- label
function CtcCriterion:_getLabels(target)
    local labels = {}
    local seQCount = target:size(1) / self.history
    -- initialize and insert first label
    for i=1, seQCount do
        table.insert(labels, {target[i]})
    end

    -- iterate over labels and if target is not the same as in the previous
    -- timestep insert it
    for i=1, self.history - 1 do
        for y=1, seQCount do
            if target[i * seQCount + y] ~= target[(i-1) * seQCount + y] then
                table.insert(labels[y], target[i * seQCount + y])
            end
        end
    end
    return labels
end


function CtcCriterion:updateGradInput(input, target)
    return self.exp:backward(input, self.grads)
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
