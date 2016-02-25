require 'torch'
require 'nn'
require 'LstmSteps'
require 'BatchRecurrent'


local Lstm = torch.class('nn.Lstm', 'nn.BatchRecurrent')


function Lstm:_setActualModule()
    self.aModule = nn.LstmSteps(self.layerSize, self.history)
end


--eof
