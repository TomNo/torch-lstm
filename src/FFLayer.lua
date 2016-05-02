require 'nn'
require 'torch'

local FFLayer, parent = torch.class("nn.FFLayer", "nn.Sequential")


function FFLayer:__init(iSize, oSize, aType, bn, dropout)
    parent.__init(self)
    bn = bn or false
    dropout = dropout or 0
    self:add(nn.Linear(iSize, oSize))
    if bn then
        self:add(nn.BatchNormalization(oSize))
    end
    if dropout ~= 0 then
        self:add(nn.Dropout(dropout))
    end
    self:add(aType())
end

--eof
