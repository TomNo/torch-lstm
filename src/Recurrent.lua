require 'torch'
require 'nn'


local Recurrent, Steps = torch.class("nn.Recurrent", "nn.Steps")


function Recurrent:__init(act, layerSize, history)
    self.activation = act
    Steps.__init(layerSize, history)
end

function Recurrent:_set