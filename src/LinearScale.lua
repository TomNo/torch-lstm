require 'torch'
require 'nn'

-- Class for scaling peepholes
local LinearScale = torch.class('LinearScale', 'nn.Linear')


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
        self.output:cmul(input, self.weight:repeatTensor(input:size(1), 1))
    else
        error('Input must be vector or matrix')
    end
    return self.output
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
            self.gradInput:cmul(gradOutput, self.weight:repeatTensor(gradOutput:size(1), 1))
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

return LinearScale