require 'torch'
require 'nn'
require 'Steps'
require 'LstmStep'


local LstmSteps = torch.class("nn.LstmSteps", "nn.Steps")

function LstmSteps:_setStepModule()
    self.step = nn.LstmStep(self.layerSize)
end


--eof
