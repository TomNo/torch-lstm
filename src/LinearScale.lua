require 'torch'
require 'nn'

--[[
-- LinearScale performs peephole scaling operation in LSTM architecture
--]]


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
    self.output = input[1]
    input = input[2]
    if input:dim() == 1 then
        error("Only batch mode is supported.")
    elseif input:dim() == 2 then
        -- TODO add some defensive coding
        self.output:resizeAs(input)
        self.output:addcmul(input, self:viewWeights(input:size(1)))
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
    input = input[2]
    if self.gradInput then
        if input:dim() == 1 then
            error("Only batch mode is supported.")
        elseif input:dim() == 2 then
            self.gradInput:resizeAs(input)
            self.gradInput:cmul(gradOutput, self:viewWeights(gradOutput:size(1)))
        end

        return self.gradInput
    end
end


function LinearScale:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    input = input[2]
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
