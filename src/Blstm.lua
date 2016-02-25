require 'torch'
require 'nn'
require 'LstmSteps'
require 'Bidirectional'


local Blstm = torch.class('nn.Blstm', 'nn.Bidirectional')


function Blstm:_setActualModule()
    self.aModule = nn.LstmSteps(self.aSize, self.history)
end

--eof
