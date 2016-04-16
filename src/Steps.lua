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
        self:add(self.step:clone('weight', 'bias', 'gradWeight', 'gradBias', 'gradInput'))
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
    self.batchSize = input:size(1) / self.history
    self.output:resize(input:size(1), self.step.layerSize)
    self.output:zero()
    local maxT = self.sizes[1]
    for i = 1, maxT do
        local step = self.modules[i]
        local interval
        if not self.revert then
            interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        else
            interval = { { (maxT - i) * self.batchSize + 1, (maxT - i + 1) * self.batchSize } }
        end
        step:forward(input[interval])
        if self.revert then
            for s=1,#self.sizes do
                if self.history - self.sizes[s] - 1> i then
                    step.output[s]:zero()
                end
            end
        else
            for s=1,#self.sizes do
                if self.sizes[s] < i then
                    step.output[s]:zero()
                end
            end
        end

        self.output[interval]:copy(step.output)
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
    self.gradInput:zero()
    local lModule = self.modules[self.sizes[1]]
    local bNStep = lModule.nStep
    lModule.nStep = nil
    local maxT = self.sizes[1]
    for i = maxT, 1, -1 do
        local interval
        if not self.revert then
            interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        else
            interval = { { (maxT - i) * self.batchSize + 1, (maxT - i + 1) * self.batchSize } }
        end
        local step = self.modules[i]
        step:backward(input[interval], gradOutput[interval])
        self.gradInput[interval]:copy(step:getGradInput())
    end
    lModule.nStep = bNStep
    return self.gradInput
end

--eof
