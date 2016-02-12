require 'torch'
require 'nn'
require 'Lstm'
require 'Revert'


local Blstm = torch.class('nn.Blstm', 'nn.Sequential')

local InputActs = torch.class('InputActs', 'nn.Linear')


function InputActs:updateOutput(input)
  nn.Linear.updateOutput(self, input)
  local size = self.output:size(2)
  self.output = {self.output[{{}, {1, size/2}}],
                 self.output[{{}, {size/2 + 1, size}}]}
  return self.output
end

function InputActs:backward(input, gradOutput)
  gradOutput = torch.cat(gradOutput)
  nn.Module.backward(self, input, gradOutput)
  return self.gradInput
end


function Blstm:__init(inputSize, layerSize, hist, bNorm)
  assert(layerSize % 2 == 0, "Layer must have even count of neurons.")
  nn.Sequential.__init(self)
  local oSize = layerSize / 2
  self.bNorm = bNorm or false
  self.layerSize = layerSize
  self.inputSize = inputSize
  self.history = hist
  self.iActs = InputActs.new(inputSize, self.layerSize * 4)
  self:add(self.iActs) -- input activations
  self.f_lstm = nn.LstmSteps()


  local concat = nn.ConcatTable()
--  self.b_lstm = nn.Sequential():add(Revert.new()):add(Lstm.new(inputSize, o_size, hist, self.b_norm)):add(Revert.new())
  self.b_lstm = nn.Sequential():add(nn.Lstm(inputSize, o_size, hist, self.b_norm))
  self.f_lstm = nn.Lstm(inputSize, o_size, hist, self.b_norm)
  concat:add(self.b_lstm):add(self.f_lstm)
  self.module:add(concat)
  self.module:add(nn.JoinTable(2))
  self.modules = {self.module}
end

--function Blstm:updateGradInput(input, gradOutput)
--  local f_gradOutput = gradOutput[{{}, {1, self.layerSize / 2}}]
--  local b_gradOutput = gradOutput[{{}, {self.layerSize / 2 + 1 , self.layerSize}}]
--  self.gradInput:resize(self.batch_size * self.history_size, self.inputSize)
--  self.f_lstm:backward(input, f_gradOutput)
--  self.b_lstm:backward(self.r_input, b_gradOutput)
--  self.gradInput = self.f_lstm.gradInput + self.b_lstm.gradInput  
--  return self.gradInput
--end
--
--function Blstm:updateOutput(input)
--  self.batch_size = input:size(1) / self.history_size
--  local r_input = self:revertInput(input)
--  self.output = torch.cat(self.f_lstm:forward(input), self.b_lstm:forward(self.r_input))
--  return self.output  
--end

function Blstm:backward(input, gradOutput)
  self.gradInput =  self.module:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Blstm:updateOutput(input)
  return self.module:updateOutput(input)
end

function Blstm:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d, BatchNormalized=%s)', self.inputSize, self.layerSize, self.b_norm)
end

--a = Lstm.new(10, 20, 50)
--a:training()
--print(a.model)
--inp = torch.randn(16*50,10)
--a:evaluate()
--output = a(inp)
--a:backward(inp, torch.randn(16*50, 20))

function testBlstm()
  local c = nn.Blstm(2, 2, 1)
  local b = nn.Blstm(2,2,1)
  local a = nn.Sequential()
  a:add(b)
  a:add(c)
  print(a:forward(torch.ones(2,2)))
  print(a:backward(torch.ones(2,2), torch.ones(2,2)))
end

--testBlstm()
