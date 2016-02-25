require 'torch'
require 'nn'
require 'GruSteps'


-- implemented according to http://arxiv.org/pdf/1412.3555v1.pdf

local Gru = torch.class('nn.Gru', 'nn.Sequential')


function Gru:__init(inputSize, layerSize, hist, bNorm)
    nn.Sequential.__init(self)
    --module for computing all input activations
    local aCount = 3 * layerSize
    self.layerSize = layerSize
    self.inputSize = inputSize
    -- set biases for all units in here -> temporary to one
    self.iActsModule = nn.Linear(inputSize, aCount)
    self.iActsModule.bias:fill(1)
    self:add(self.iActsModule)
    self.bNorm = bNorm or false
    if self.bNorm then
        self:add(nn.BatchNormalization(aCount))
    end
    self:add(nn.GruSteps(layerSize, hist))
end


function Gru:__tostring__()
    return torch.type(self) ..
            string.format('(%d -> %d, BatchNormalized=%s)', self.inputSize,
                self.layerSize,
                self.bNorm)
end


--eof
