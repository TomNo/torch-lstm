require 'torch'
require 'nn'
require 'GruSteps'
require 'Bidirectional'

--[[
-- Bidirectional gated recurrent nit
 ]]

local Bgru = torch.class('nn.Bgru', 'nn.Bidirectional')


function Bgru:_setActualModule()
    self.aModule = nn.GruSteps(self.aSize, self.history)
end


--eof
