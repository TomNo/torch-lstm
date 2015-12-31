require 'torch'
require 'cutorch'
require 'nn'

local Lstm, parent = torch.class('Lstm', 'nn.Container')

--TODO there is no support for non batch mode

function Lstm:__init(inputSize, outputSize, b_size, hist)
  parent.__init(self)
  self.p = torch.Tensor(1)
  self.batch_size = b_size
  self.history_size = hist -- history size
  self.outputSize = outputSize
  self.c_states = torch.Tensor() -- history of cell states
  self.g_outputs = torch.Tensor() -- grad output from step modules
  self.i_acts = nil -- input activations
  --module for computing all input activations
  self.a_count = 4 * outputSize
  local p_count = 2 * outputSize
  self.i_acts_module = nn.Linear(inputSize, self.a_count)
  table.insert(self.modules, self.i_acts_module)
  --module for computing one mini batch
  self.model = nn.Sequential()
  local i_acts = nn.Identity()
  -- all output activations
  local o_acts = nn.Linear(outputSize,self.a_count)
  -- forget and input peepholes cell acts
  local c_acts = nn.ConcatTable():add(nn.Linear(inputSize, p_count)):add(nn.Identity())
  -- container for summed input and output activations
  -- that is divided in half
  local io_acts = nn.Sequential()
  io_acts:add(nn.ParallelTable():add(i_acts):add(o_acts))
  io_acts:add(nn.CAddTable()):add(nn.Reshape(2, 2 * outputSize)):add(nn.SplitTable(1,2))
  -- sum half of the activations with peepholes
  self.model:add(nn.ParallelTable():add(io_acts):add(c_acts))
  self.model:add(nn.FlattenTable())
  -- output of the model at this stage is <c_states + o_acts, i_acts + f_acts, peepholes acts, cell states>
  -- input and forget gate activation
  local gates = nn.ConcatTable()
  gates:add(nn.Sequential():add(nn.NarrowTable(2,2)):add(nn.CAddTable()):add(nn.Sigmoid()):add(nn.Reshape(2, outputSize)):add(nn.SplitTable(1,2)))
  gates:add(nn.Sequential():add(nn.SelectTable(4)))
  --  -- divide rest activations between cell state and output gate
  gates:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Reshape(2, outputSize)):add(nn.SplitTable(1,2)))
  self.model:add(gates)
  self.model:add(nn.FlattenTable())
  -- output of the model at this stage is <i_acts, f_acts, cell states, c_acts, o_acts>
  local cell_acts = nn.ConcatTable()
  -- forward i_acts
  cell_acts:add(nn.SelectTable(1))
  -- apply squashing function to cell state
  cell_acts:add(nn.Sequential():add(nn.SelectTable(4)):add(nn.Tanh()))
  -- apply forgeting
  cell_acts:add(nn.Sequential():add(nn.NarrowTable(2,2)):add(nn.CMulTable()))
  -- forward o_acts
  cell_acts:add(nn.SelectTable(5))
  -- output of the model at this stage is <i_acts, c_acts, f_acts, o_acts>
  self.model:add(cell_acts)
  cell_acts = nn.ConcatTable()
  -- scale cell state by input
  cell_acts:add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CMulTable()))
  -- forward
  cell_acts:add(nn.Sequential():add(nn.SelectTable(3)))
  cell_acts:add(nn.Sequential():add(nn.SelectTable(4)))
  -- output of the model at this stage is <c_acts, f_acts, o_acts>
  -- add previous cell state
  self.model:add(cell_acts)
  cell_acts = nn.ConcatTable()
  cell_acts:add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
  cell_acts:add(nn.Sequential():add(nn.SelectTable(3)))
  self.model:add(cell_acts)
  -- output of the model at this stage is <c_acts, o_acts>
  -- scale by peephole from the cell state to output gate
  cell_acts = nn.ConcatTable()
  cell_acts:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Linear(outputSize, outputSize)))
  cell_acts:add(nn.SelectTable(2))
  self.model:add(cell_acts)
  self.model:add(nn.NarrowTable(1,2)):add(nn.CMulTable())
  -- result is <output>

  -- copies of 'step' module
  table.insert(self.modules, self.model)
  for i=2, hist do
    table.insert(self.modules, self.model:clone('weight','bias'))
  end
end

function Lstm:updateGradInput(input, gradOutput)
  self.g_outputs:resize(self.batch_size * self.history_size, self.a_count)
  for i=2, #self.modules do
    local step = self.modules[i]
    local s = (i-2)*self.batch_size + 1
    local e = (i-1)*self.batch_size
    local bck = step:backward(self.i_acts[{{s,e},{}}], gradOutput[{{s,e}, {}}])
    self.g_outputs[{{s,e}, {}}]:copy(bck[1])
  end
  self.i_acts_module:backward(input, self.g_outputs)
end

function Lstm:updateOutput(input)
  -- TODO this resizing might be handled better
  self.output:resize(self.history_size * self.batch_size, self.outputSize)
  self.c_states:resize(self.history_size * self.batch_size, self.outputSize)
  self.i_acts = self.i_acts_module:forward(input)
  -- do first step manually, set previous output and previous cell state to zeros
  local z_tensor = torch.zeros(self.batch_size, self.outputSize)
  self.model:forward({{i_acts[{{1, self.batch_size}, {}}], z_tensor}, z_tensor})
  self.output[{{1, self.batch_size}, {}}]:copy(self.model.output)
  for i= 3, #self.modules do
    local p_step = self.modules[i-1]
    local step = self.modules[i]
    local s = (i-2)*self.batch_size + 1
    local e = (i-1)*self.batch_size
    local t_i_acts = i_acts[{{s, e},{}}]
    step:forward({{t_i_acts, p_step.output}, self:getCellStates(p_step)})
    self.output[{{s,e},{}}]:copy(step.output)
  end
  return self.output
end

function Lstm:getCellStates(model)
  -- gathers cell states from particular module
  return model:get(7).output[1]
end
