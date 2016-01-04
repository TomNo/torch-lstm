require 'torch' 
require 'cutorch'
require 'nn'

local NonBiasLinear, p_linear, p_module = torch.class('NonBiasLinear', 'nn.Linear', 'nn.Module')

-- just remove everything about bias from nn.Linear class
function NonBiasLinear:__init(inputSize, layerSize)
   p_module.__init(self)
   self.layerSize = layerSize
   self.weight = torch.Tensor(layerSize, inputSize)
   self.gradWeight = torch.Tensor(layerSize, inputSize)
   self:reset()
end

function NonBiasLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end
   return self
end

function NonBiasLinear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.layerSize)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end
   return self.output
end

function NonBiasLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end

-- we do not need to accumulate parameters when sharing
NonBiasLinear.sharedAccUpdateGradParameters = NonBiasLinear.accUpdateGradParameters


