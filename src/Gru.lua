require 'torch'
require 'nn'
require 'BatchRecurrent'
require 'GruSteps'

--[[
-- Gated recurrent unit,
-- implemented according to http://arxiv.org/pdf/1412.3555v1.pdf
 ]]

local Gru = torch.class('nn.Gru', 'nn.BatchRecurrent')


function Gru:_setActualModule()
    self.aModule = nn.GruSteps(self.layerSize, self.history)
end


--eof
