require 'torch'
require 'nn'
require 'LstmSteps'
require 'BatchRecurrent'


--[[
-- Lstm class,
-- implementation based on http://www.cs.toronto.edu/~graves/phd.pdf
 ]]


local Lstm = torch.class('nn.Lstm', 'nn.BatchRecurrent')


function Lstm:_setActualModule()
    self.aModule = nn.LstmSteps(self.layerSize, self.history)
end


--eof
