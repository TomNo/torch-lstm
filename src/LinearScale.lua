require 'torch'
require 'nn'

-- Class for scaling peepholes
local LinearScale = torch.class('nn.LinearScale', 'nn.Linear')


function LinearScale:__init(outputSize)
    nn.Module.__init(self)
    self.weight = torch.Tensor(outputSize)
    self.gradWeight = torch.Tensor(outputSize)
    self:reset()
end


function LinearScale:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1. / math.sqrt(self.weight:size(1))
    end
    self.weight:uniform(-stdv, stdv)

    return self
end


function LinearScale:updateOutput(input)
    if input:dim() == 1 then
        self.output:resizeAs(input)
        self.output:copy(input)
        self.output:cmul(self.weight)
    elseif input:dim() == 2 then
        self.output:resizeAs(input)
        self.output:cmul(input, self:viewWeights(input:size(1)))
    else
        error('Input must be vector or matrix')
    end
    return self.output
end


function LinearScale:viewWeights(size)
    local tmp = self.weight:view(1, self.weight:size(1))
    return tmp:expand(size, self.weight:size(1))
end


function LinearScale:updateGradInput(input, gradOutput)
    if self.gradInput then
        local nElement = self.gradInput:nElement()
        self.gradInput:resizeAs(input)
        if self.gradInput:nElement() ~= nElement then
            self.gradInput:zero()
        end
        if input:dim() == 1 then
            self.gradInput:addcmul(self.weight, gradOutput)
        elseif input:dim() == 2 then
            self.gradInput:cmul(gradOutput, self:viewWeights(gradOutput:size(1)))
        end

        return self.gradInput
    end
end


function LinearScale:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if input:dim() == 1 then
        self.gradWeight:addcmul(input, gradOutput)
    else
        self.gradWeight:add(torch.cmul(input, gradOutput):sum(1))
    end
    self.gradWeight:mul(scale)
end


function LinearScale:__tostring__()
    return torch.type(self) ..
            string.format('(%d -> %d)', self.weight:size(1),
                self.weight:size(1))
end

--eof
