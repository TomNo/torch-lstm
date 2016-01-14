require 'torch'
require 'cutorch'
require 'nn'
require 'lstm'


local Blstm, parent = torch.class('Blstm', 'nn.Container')

--just spawn two lstms that will process output in oposite direction

function Blstm:__init(inputSize, layerSize, hist)
  assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
  parent.__init(self)
  local o_size = layerSize / 2;
  self.b_norm = b_norm or false
  self.layerSize = layerSize
  self.inputSize = inputSize
  self.history_size = hist
  self.b_lstm = Lstm.new(inputSize, o_size, hist)
  self.f_lstm = Lstm.new(inputSize, o_size, hist)
  self.modules = {self.f_lstm, self.b_lstm}
  self.r_input = torch.Tensor()
end

function Blstm:updateGradInput(input, gradOutput)
  local f_gradOutput = gradOutput[{{}, {1, self.layerSize / 2}}]
  local b_gradOutput = gradOutput[{{}, {self.layerSize / 2 + 1 , self.layerSize}}]
  self.gradInput:resize(self.batch_size * self.history_size, self.inputSize)
  self.f_lstm:backward(input, f_gradOutput)
  self.b_lstm:backward(self.r_input, b_gradOutput)
  self.gradInput = self.f_lstm.gradInput + self.b_lstm.gradInput  
  return self.gradInput
end

function Blstm:revertInput(input)
  self.r_input:resizeAs(input)
  for i=1,input:size(1) do
    self.r_input[i] = input[input:size(1) + 1 - i]
  end
  return self.r_input
end

function Blstm:updateOutput(input)
  self.batch_size = input:size(1) / self.history_size
  self:revertInput(input)
  self.output = torch.cat(self.f_lstm:forward(input), self.b_lstm:forward(self.r_input))
  return self.output  
end

function Blstm:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.inputSize, self.layerSize)
end

--a = Lstm.new(10, 20, 50)
--a:training()
--print(a.model)
--inp = torch.randn(16*50,10)
----a:evaluate()
--output = a(inp)
--a:backward(inp, torch.randn(16*50, 20))

return Blstm


