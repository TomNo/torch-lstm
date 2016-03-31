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
    self.revert = false -- take input in reverse order - for bidirectional
    -- copies of first step module
    self:add(self.step)
    for _ = 2, self.history do
        self:add(self.step:clone('weight', 'bias', 'gradWeight', 'gradBias'))
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


function Steps:updateGradInput(...)
    error("This method should not be used within this class.")
end


function Steps:accGradParameters(...)
    error("This method should not be used withing this class.")
end


function Steps:updateOutput(input)
    self.batchSize = input[1]:size(1) / self.history
    self.output:resize(input[1]:size(1), self.step.layerSize)
    for i = 1, input[1]:size(1) / self.batchSize do
        local step = self.modules[i]
        local interval
        if not self.revert then
            interval = { { (i - 1) * self.batchSize + 1, i * self.batchSize } }
        else
            interval = { { (self.history - i) * self.batchSize + 1, (self.history - i + 1) * self.batchSize } }
        end

        local aInput
        if not torch.isTensor(input) then
            aInput = {}
            for i=1,#input do
                table.insert(aInput, input[i][interval])
            end
        else
            aInput = input[interval]
        end
        step:forward(aInput)
        self.output[interval]:copy(step.output)
    end
    return self.output
end


--eof
