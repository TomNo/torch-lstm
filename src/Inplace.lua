require 'nn'
require 'torch'


nn.Sigmoid.__init = function(self, inplace)
    self.inplace = inplace
    nn.Module.__init(self)
end

nn.Sigmoid.updateOutput = function(self, input)
    if self.inplace then
        self.output = input
    end
        input.THNN.Sigmoid_updateOutput(
        input:cdata(),
        input:cdata())
   return self.output
end


nn.Tanh.__init = function(self, inplace)
    self.inplace = inplace
    nn.Module.__init(self)
end


nn.Tanh.updateOutput = function(self, input)
    if self.inplace then
        self.output = input
    end
        input.THNN.Tanh_updateOutput(
        input:cdata(),
        input:cdata())
   return self.output
 end