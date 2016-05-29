require 'utils'
require 'torch'
require 'nn'
require 'LstmStep'
require 'Module'


local Steps = torch.class("nn.Steps", "nn.Container")


function Steps:__init(layerSize, history)
    nn.Container.__init(self)
    self.mask = torch.Tensor()
    self.history = history or 1
    self.layerSize = layerSize
    self:_setStepModule()
    self.inputSize = self.step.inputSize
    self.revert = false
    -- copies of first step module
    self:add(self.step)
    for _ = 2, self.history do
        self:add(self.step:clone('weight', 'bias', 'gradWeight', 'gradBias',
            'gradInput'))
    end

    -- set to every module which module is next and previous
    for i = 1, self.history do
        local step = self.modules[i]
        if i > 1 then step.pStep = function() return self.modules[i - 1] end end
        if i < #self.modules then step.nStep = function() return self.modules[i + 1] end end
    end
end


function Steps:_setStepModule()
    error("This method must be implemented by superclass.")
end


function Steps:updateOutput(input)
    self.output:resize(input:size(1), self.step.layerSize)
    local ptr = 0
    if self.revert then
        ptr = input:size(1)
    end
    local maxT = #self.bSizes
    for i = 1, #self.bSizes do
        local step = self.modules[i]
        local interval
        if not self.revert then
            interval = { { ptr + 1, ptr + self.bSizes[i] } }
        else
            interval = {{ptr - self.bSizes[maxT - i + 1] + 1, ptr}}
        end
        step:forward(input[interval])
        self.output[interval]:copy(step.output)
        if self.revert then
            ptr = ptr - self.bSizes[maxT - i + 1]
        else
            ptr = ptr + self.bSizes[i]
        end
    end
    return self.output
end


function Steps:updateGradInput(input, gradOutput)
    error("This method should not be used, use backward instead.")
end


function Steps:accGradParameters(input, gradOutput)
    error("This method should not be used, use backward instead.")
end


function Steps:backward(input, gradOutput)
    self.gradInput:resizeAs(input)
    local ptr = 0
    if not self.revert then
        ptr = utils.sumTable(self.bSizes)
    end
    local lModule = self.modules[#self.bSizes]
    local nStepBck = lModule.nStep
    lModule.nStep = nil
    for i = #self.bSizes, 1, -1 do
        local interval
        if not self.revert then
            interval = { { ptr - self.bSizes[i] + 1, ptr } }
        else
            interval = {{ptr + 1, ptr + self.bSizes[#self.bSizes - i + 1]}}
        end
        local step = self.modules[i]
        step:backward(input[interval], gradOutput[interval])
        self.gradInput[interval]:copy(step:getGradInput())
        if not self.revert then
            ptr = ptr - self.bSizes[i]
        else
            ptr = ptr + self.bSizes[#self.bSizes - i + 1]
        end
    end
    lModule.nStep = nStepBck
    return self.gradInput
end

--eof
