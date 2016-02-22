require 'torch'
require 'nn'
require 'Revert'
require 'Split'
require 'LstmStep'
require 'Steps'


local Blstm = torch.class('nn.Blstm', 'nn.Sequential')


function Blstm:__init(inputSize, layerSize, hist, bNorm)
    assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
    nn.Sequential.__init(self)
    local oSize = layerSize / 2
    self.bNorm = bNorm or false
    self.layerSize = layerSize
    self.inputSize = inputSize
    self.history = hist
    self:add(nn.Linear(inputSize, self.layerSize * 4)) -- input activations
    if bNorm then
        self:add(nn.BatchNormalization(self.layerSize * 4))
    end
    self:add(nn.Split())
    local fStep = nn.LstmStep(oSize, bNorm)
    local bStep = nn.LstmStep(oSize, bNorm)
    self.fLstm = nn.Steps(fStep, hist)
    self.bLstm = nn.Sequential()
    self.bLstm:add(nn.Revert())
    self.bLstm:add(nn.Steps(bStep, hist))
    self.bLstm:add(nn.Revert())
    local pTable = nn.ParallelTable()
    pTable:add(self.fLstm)
    pTable:add(self.bLstm)
    self:add(pTable)
    self:add(nn.JoinTable(2, 2))
end


function Blstm:__tostring__()
    return torch.type(self) ..
            string.format('(%d -> %d, BatchNormalized=%s)', self.inputSize, self.layerSize, self.bNorm)
end

--a = Lstm.new(10, 20, 50)
--a:training()
--print(a.model)
--inp = torch.randn(16*50,10)
--a:evaluate()
--output = a(inp)
--a:backward(inp, torch.randn(16*50, 20))

--function testBlstm()
--  local c = nn.Blstm(2, 2, 1)
--  local b = nn.Blstm(2,2,1)
--  local a = nn.Sequential()
--  a:add(b)
--  a:add(c)
--  print(c:forward(torch.ones(2,2)))
--  print(c:backward(torch.ones(2,2), torch.ones(2,2)))
--end
--
--testBlstm()
