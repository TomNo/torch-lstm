require 'torch'
require 'nn'
require 'LstmStep'


local LstmSteps = torch.class("nn.LstmSteps", "nn.Container")

function LstmSteps:__init(layerSize, bNorm, history)
    nn.Container.__init(self)
    self.history = history or 1
    self.layerSize = layerSize
    --module for computing one mini batch in one timestep
    local fStep = nn.LstmStep(layerSize, bNorm)
    -- copies of first step module
    self:add(fStep)
    for _ = 2, self.history do
        self:add(fStep:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    end

    -- set to every module which module is next and previous
    for i = 1, self.history do
        local step = self.modules[i]
        if i > 1 then step.pStep = function() return self.modules[i - 1] end end
        if i < #self.modules then step.nStep = function() return self.modules[i + 1] end end
    end
end


function LstmSteps:updateOutput(input)
    if not self.train and input:size(1) < self.history then
        -- there can be sequence that is shorter than required history
        self.historyBackup = self.history
        self.history = input:size(1)
    end
    self.batchSize = input:size(1) / self.history
    self.output:resize(input:size(1), self.layerSize)
    for i = 1, input:size(1) / self.batchSize do
        local step = self.modules[i]
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        step:forward(input[interval])
        self.output[interval]:copy(step.output)
    end
    if not self.train and self.historyBackup then
        self.history = self.historyBackup
        self.historyBackup = nil
    end
    return self.output
end


function LstmSteps:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    for i = #self.modules, 1, -1 do
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        local step = self.modules[i]
        step:updateGradInput(input[interval], gradOutput[interval])
        self.gradInput[interval]:copy(step:getGradInput())
    end
    return self.gradInput
end


function LstmSteps:accGradParameters(input, gradOutput)
    for i = #self.modules, 1, -1 do
        local interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        local step = self.modules[i]
        step:accGradParameters(input[interval], gradOutput[interval])
    end
end


function LstmSteps:backward(input, gradOutput)
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
