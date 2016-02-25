require 'torch'
require 'nn'
require 'Steps'
require 'GruStep'


local GruSteps = torch.class("nn.GruSteps", "nn.Steps")

function GruSteps:_setStepModule()
    self.step = nn.GruStep(self.layerSize)
end


--eof
