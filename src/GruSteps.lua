require 'torch'
require 'nn'
require 'Steps'
require 'GruStep'


local GruSteps = torch.class("nn.GruSteps", "nn.Steps")

function LstmSteps:_setStepModule()
    self.step = nn.GruSteps(self.layerSize)
end


--eof
