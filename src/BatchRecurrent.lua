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
--        if self.bNorm then
--            linModule:add(nn.BatchNormalization(layerSize))
--        end
        self.sharedInput:add(linModule)
    end
    self:add(self.sharedInput)
    self:add(self.aModule)
end


function BatchRecurrent:_setActualModule()
    error("This method muset be overiden in superclass.")
end


function BatchRecurrent:backward(input, gradOutput)
    local batchSize = input:size(1) / self.history
    self.gradInput:resizeAs(input)
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
        self.sharedInput.gradInput = self.gradInput[interval]
        self.sharedInput:backward(input[interval], step:getGradInput())
    end
    return self.gradInput
end


function BatchRecurrent:__tostring__()
    return torch.type(self) .. string.format('(%d -> %d, BatchNormalized=%s)',
                                             self.inputSize,
                                             self.layerSize,
                                             self.bNorm)
end