require 'torch'
require 'nn'
require 'SharedInput'

local BatchRecurrent = torch.class('nn.BatchRecurrent', 'nn.Sequential')


function BatchRecurrent:__init(inputSize, layerSize, hist, bNorm)
    nn.Sequential.__init(self)
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history =  hist
    self.bNorm = bNorm or false
    self:_setActualModule()
    --module for computing all input activations
    self.sharedInput = nn.SharedInput()
    local iSize = self.aModule.inputSize / layerSize
    for i=1, iSize do
        local linModule = nn.Sequential()
        linModule:add(nn.Linear(inputSize, layerSize, false))
        if self.bNorm then
            linModule:add(nn.BatchNormalization(layerSize))
        end
        self.sharedInput:add(linModule)
    end
    self:add(self.sharedInput)
    self:add(self.aModule)
    self.bGradInput = self.gradInput
end


function BatchRecurrent:_setActualModule()
    error("This method muset be overiden in superclass.")
end


function BatchRecurrent:backward(input, gradOutput)
    local batchSize = input:size(1) / self.history
    local linInput = self.aModule.inputSize / self.layerSize
    self.bGradInput:resize(input:size(1)*linInput, self.layerSize)
    self.gradInput = {}
    for i=1, self.aModule.inputSize / self.layerSize do
        table.insert(self.gradInput, self.bGradInput[{{(i - 1)*input:size(1)+1,  i *input:size(1)}}])
    end

    for i = #self.aModule.modules, 1, -1 do
        local interval
        if self.aModule.revert then
            interval = { { (self.history - i) * batchSize + 1, (self.history - i) * batchSize } }
        else
            interval = { { (i - 1) * batchSize + 1, i * batchSize } }
        end
        local step = self.aModule.modules[i]
        local aInput = {}
        for i=1, #self.sharedInput.output do
            table.insert(aInput, self.sharedInput.output[i][interval])
        end
        step:backward(aInput, gradOutput[interval])
        local stepGrad = step:getGradInput()
        for y=1, #stepGrad do
            self.gradInput[y][interval]:copy(stepGrad[y])
        end
    end
    self.sharedInput:backward(input, self.gradInput)
    self.gradInput = self.bGradInput
    self.gradInput:resizeAs(self.sharedInput.gradInput)
    self.gradInput:copy(self.sharedInput.gradInput)
    return self.gradInput
end


function BatchRecurrent:__tostring__()
    return torch.type(self) .. string.format('(%d -> %d, BatchNormalized=%s)',
                                             self.inputSize,
                                             self.layerSize,
                                             self.bNorm)
end