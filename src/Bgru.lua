require 'torch'
require 'nn'
require 'GruSteps'
require 'Bidirectional'


local Bgru = torch.class('nn.Bgru', 'nn.Bidirectional')


function Bgru:_setActualModule()
    self.aModule = nn.Gru(self.inputSize, self.aSize, self.history, self.bNorm)
end


--eof
