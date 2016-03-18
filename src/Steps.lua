require 'torch'
require 'nn'
require 'LstmStep'
require 'Module'


local Steps = torch.class("nn.Steps", "nn.Container")


function Steps:__init(layerSize, history)
    nn.Container.__init(self)
    self.history = history or 1
    self.layerSize = layerSize
    self:_setStepModule()
    self.inputSize = self.step.inputSize
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
    for i = 1, input:size(1) / self.batchSize do
        local step = self.modules[i]
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        step:forward(input[interval])
        self.output[interval]:copy(step.output)
    end
    return self.output
end


function Steps:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    for i = #self.modules, 1, -1 do
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        local step = self.modules[i]
        step:updateGradInput(input[interval], gradOutput[interval])
        self.gradInput[interval]:copy(step:getGradInput())
    end
    return self.gradInput
end


function Steps:accGradParameters(input, gradOutput)
    for i = #self.modules, 1, -1 do
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        local step = self.modules[i]
        step:accGradParameters(input[interval], gradOutput[interval])
    end
end


function Steps:backward(input, gradOutput)
    self.gradInput:resizeAs(input)
    for i = #self.modules, 1, -1 do
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        local step = self.modules[i]
        step:backward(input[interval], gradOutput[interval])
        self.gradInput[interval]:copy(step:getGradInput())
    end
    return self.gradInput
end

--eof
