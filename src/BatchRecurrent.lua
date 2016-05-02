require 'torch'
require 'nn'


local BatchRecurrent = torch.class('nn.BatchRecurrent', 'nn.Sequential')


function BatchRecurrent:__init(inputSize, layerSize, hist, bNorm, dropout)
    nn.Sequential.__init(self)
    self.dropout = dropout or 0
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history =  hist
    self.bNorm = bNorm or false
    self:_setActualModule()
    --module for computing all input activations
    self.iActsModule = nn.Linear(inputSize, self.aModule.inputSize, false)
    self:add(self.iActsModule)
    if self.bNorm then
        self:add(nn.BatchNormalization(self.aModule.inputSize))
    end
    if self.dropout ~= 0 then
        self:add(nn.Dropout(self.dropout))
    end
    self:add(self.aModule)
end


function BatchRecurrent:_setActualModule()
    error("This method muset be overiden in superclass.")
end


function BatchRecurrent:__tostring__()
    return torch.type(self) .. string.format('(%d -> %d, BatchNormalized=%s)',
                                             self.inputSize,
                                             self.layerSize,
                                             self.bNorm,
                                             self.dropout)
end