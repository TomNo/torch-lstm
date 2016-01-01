require 'torch'
require 'cutorch'
require 'nn'
require 'lstm'

local Blstm, parent = torch.class('Blstm', 'nn.Container')

--just spawn two lstms that will process output in oposite direction

function Blstm:__init(inputSize, layerSize, b_size, hist)
  assert(inputSize % 2 == 0, "Layer must have even count of neurons.")
  parent.__init(self)
  local i_size = inputSize / 2;
  local o_size = layerSize / 2;
  self.layerSize = layerSize
  self.intputSize = inputSize
  self.batch_size = b_size
  self.history_size = hist
  self.b_lstm = lstm.Lstm.new(i_size, o_size, b_size, hist)
  self.f_lstm = lstm.Lstm.new(i_size, o_size, b_size, hist)
  self.modules = {self.f_lstm, self.b_lstm}
  self.r_input = torch.Tensor()
end

function Blstm:updateGradInput(input, gradOutput)
  local f_gradOutput = gradOutput[{{1, #gradOutput/2}, {}}]
  local b_gradOutput = gradOutput[{{#gradOutput/2, #gradOutput}, {}}]
  self.gradInput:resize(self.batch_size * self.history_size, self.inputSize)
  self.f_lstm:backward(input, f_gradOutput)
  self.b_lstm:backward(self.r_input, b_gradOutput)
  self.gradInput = self.f_lstm.gradInput + self.b_lstm.gradInput  
  return self.gradInput
end

function Blstm:revertInput(input)
  self.r_input:resizeAs(input)
  for i=1,#input do
    self.r_input[i] = input[#input + 1 - i]
  end
  return self.r_input
end

function Blstm:updateOutput(input)
  self:revertInput(input)
  self.output = torch.cat(self.f_lstm:forward(input), self.b_lstm:forward(self.r_input))
  return self.output  
end

--a = Lstm.new(10, 20, 16, 50)
--print(a.model)
--inp = torch.randn(16*50,10)
--output = a(inp)
--a:backward(inp, torch.randn(16*50, 20))

