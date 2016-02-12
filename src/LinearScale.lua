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


function LinearScale:__tostring__()
    return torch.type(self) ..
        string.format('(%d -> %d)', self.weight:size(1),
                                    self.weight:size(1))
end


function testOutput()
    local layerSize = 5
    local a = nn.LinearScale(layerSize)
    local weights, _ = a:getParameters()
    local w_val = 0.3
    weights:fill(w_val)
    local output = a:forward(torch.ones(layerSize)):eq(w_val):sum()
    if output ~= layerSize then
        error("LinearScale output is not correct.")
    end
    local bOutput = a:forward(torch.ones(5, layerSize)):eq(w_val):sum()
    if  bOutput ~= 5*layerSize then
        error("LinearScale batched output is not correct.")
    end
end


function testBackward()
    local layerSize = 5
    local b = nn.LinearScale(layerSize)
    local weigths, g_weights = b:getParameters()
    g_weights:fill(0)
    local w_val = 0.3
    weigths:fill(w_val)
    local err = b:backward(torch.ones(layerSize), torch.ones(layerSize):fill(w_val))
    if err:eq(w_val*w_val):sum() ~= layerSize then
        error("LinearScale backward error does not fit.")
    end
    if g_weights:eq(w_val):sum() ~= layerSize then
        error("LinearScale gradients are not correct.")
    end
end


testOutput()
testBackward()
