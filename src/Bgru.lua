require 'torch'
require 'nn'
require 'Revert'
require 'Split'
require 'LstmStep'
require 'Steps'


local Bgru = torch.class('nn.Bgru', 'nn.Sequential')


function Bgru:__init(inputSize, layerSize, hist, bNorm)
    assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
    nn.Sequential.__init(self)
    local oSize = layerSize / 2
    self.bNorm = bNorm or false
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history = hist
    -- no bias in input activations
    self:add(nn.Linear(inputSize, self.layerSize * 4, false))
    if bNorm then
        self:add(nn.BatchNormalization(self.layerSize * 4))
    end
    self:add(nn.Split())
    self.fLstm = nn.GruSteps(oSize, hist)
    self.bLstm = nn.Sequential()
    self.bLstm:add(nn.Revert())
    self.bLstm:add(nn.Steps(oSize, hist))
    self.bLstm:add(nn.Revert())
    local pTable = nn.ParallelTable()
    pTable:add(self.fLstm)
    pTable:add(self.bLstm)
    self:add(pTable)
    self:add(nn.JoinTable(2, 2))
end


function Bgru:__tostring__()
    return torch.type(self) ..
            string.format('(%d -> %d, BatchNormalized=%s)', self.inputSize,
                self.layerSize, self.bNorm)
end
