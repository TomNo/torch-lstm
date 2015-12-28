require 'torch' 
require 'cutorch'
require 'nn'


-- function for transfering data to gpu memory
local function transfer_to_gpu(data)
  for _, i in ipairs(data) do
    i:cuda()
  end
end

local Lstm, parent = torch.class('Lstm', 'nn.Module')
local BiLstm, b_parent = torch.class('BiLstm', 'Lstm')


function Lstm:__init(inputSize, outputSize)
  --module for computing all input activations
  local a_count = 4 * outputSize 
  local p_count = 2 * outputSize
  self.a_i_acts = nn.Linear(inputSize, a_count)
  --module for computing one mini batch
  self.model = nn.Sequential()
  local i_acts = nn.Identity()
  -- all output activations
  local o_acts = nn.Linear(outputSize, a_count)
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
  cell_acts:add(nn.SelectTable(1))
  self.model:add(cell_acts)
  self.model:add(nn.ConcatTable():add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CMulTable())):add(nn.SelectTable(1)))
end

function Lstm:updateOutput(input)
  local i_acts = self.a_i_acts

end

function Lstm:testMe()
  print(self.model)
  print(self.model({{self.a_i_acts(torch.randn(10)), torch.randn(10)}, torch.randn(10)}))

end

a = Lstm.new(10,10)
a:testMe()

--function Lstm:__init(inputSize, outputSize)
--  local m_batch_size = 0
--  self.model = nn.Sequential()
--  local m_par_table = nn.ParallelTable()
--  -- all input activations
--  local i_acts = nn.Linear(inputSize, 4*outputSize)
--  -- all output activations
--  local o_acts = nn.Linear(outputSize, 4*outputSize)
--  -- forget and input peepholes cell acts
--  local c_acts = nn.Linear(inputSize, 2*outputSize)
--  -- container for summed input and output activations
--  local io_acts = nn.Sequential()
--  local par_table = nn.ParallelTable()
--  par_table:add(i_acts, o_acts)
--  io_acts:add(par_table)
--  io_acts:add(nn.CAddTable())
--  -- select half of the activation and sum them with peepholes
--  local concat_table = nn.ConcatTable()
--  concat_table:add(nn.NarrowTable(1, inputSize / 2))
--  concat_table:add(nn.NarrowTable(inputSize/2 + 1, inputSize/2))
--  io_acts:add(concat_table)
--  m_par_table:add(io_acts)
--  m_par_table:add(c_acts)
--  self.model:add(m_par_table)
--  local c_table = nn.ConcatTable()
--  -- selecting cell activations and output gate activations
--  c_table:add(nn.SelectTable(1))
--  -- selecting input + forget gates acts and peepholes acts
--  c_table:add(nn.NarrowTable(2,2))
--  self.model:add(c_table)
--  local p_table = nn.ParallelTable()
--  -- spliting cell activations and output gates activations
--  c_table = nn.ConcatTable()
--  c_table:add(nn.NarrowTable(1, inputSize/4))
--  c_table:add(nn.NarrowTable(inputSize/4 + 1, inputSize/4))
--  p_table:add(c_table)
--  local seq = nn.Sequential()
--  seq:add(nn.CSumTable())
--  seq:add(nn.Sigmoid())
--  c_table = nn.ConcatTable()
--  c_table:add(1, inputSize/4)
--  c_table:add(inputSize/4 + 1, inputSize/4)
--  seq:add(c_table)
--    -- input and forget gates activations done
--  p_table:add(seq)
--  self.model:add(p_table)
  
  
  
  
  
  
  
--  local ph_acts = nn.Squential():add(nn.Reshape(inputSize, 1))
--  local par_ph_acts = nn.Parallel(1, 1)
--  -- peepholes ig a fg
--  for i=1, inputSize do
--    local layer = nn.Linear(1,2)
--    layer.bias:zero()
--    par_ph_acts:add(layer)
--  end
--  o_acts.bias:zero()  -- bias is already added in the input activations
--  local par = nn.ParallelTable()
--  par:add(i_acts)
--  par:add(o_acts)
--  par:add(ph_acts)
  
  
  
  --remove bias
--  self.i_acts.bias:zero()
  
--  self.ii_g_acts = nn.Linear(inputSize, outputSize)
--  self.if_g_acts = nn.Linear(inputSize, outputSize)
--  self.ic_acts = nn.Linear(inputSize, outputSize)
--end

--local Lstm, parent = torch.class('Lstm', 'nn.Module')
--
--function Lstm:__init(params)
--  parent.__init(self)
--  for item, value in pairs(params) do
--    self[item] = value
--  end
--
--  self.outputs = torch.zeros(self.inputs:size(1), self.l_size)  
--  if params.l_size % 2 ~= 0 then
--    error("BLstm layer must have even count of neurons.")
--  end
--  
--  params.l_size = params.l_size / 2
--  
--  self.fw_layer = Lstm(params)
--  -- copy inputs and revert inputs
--  bw_params = {}
--  for item, value in pairs(params) do
--    if item ~= 'inputs' then
--      bw_params[item] = value
--    end
--  end
--  bw_inputs = torch.Tensor(params.inputs)
--  for i=1, params.inputs:size(1) do
--    bw_inputs[-i] = params.inputs[i]
--  end 
--  bw_params.inputs = bw_inputs
--  self.bw_layer = Lstm(bw_params)
--end
--
--function Lstm:forward()
--  local f_outputs = self.fw_layer:forward()
--  local b_outputs = self.bw_layer:forward()
--  local middle = self.outputs:size(2) / 2
--  for i=1,self.outputs:size(1) do
--    self.outputs[{{i},{1,middle}}] = f_outputs[i]
--    self.outputs[{{i}, {middle+1, -1}}] = b_outputs[i]
--  end
--  return self.outputs
--end

--function BiLstm:__init(i_size, o_size)
--  self.bidirectional = true
--  b_parent.__init(self, i_size, o_size)
--end
--
--Lstm.I_GATE = 1
--Lstm.F_GATE = 2
--Lstm.O_GATE = 3
--Lstm.C_STATE =4
--
--Lstm.W_T_COUNT = 4
--Lstm.P_W_COUNT = 3
--
--function Lstm:__init(i_size, o_size)
--  parent.__init(self)
----  for item, value in pairs(params) do
----    self[item] = value
----  end
--  self.i_size = i_size
--  self.o_size = o_size
--  if self.bidirectional and self.l_size % 2 ~= 0 then
--    error("Cannot consturct bidirectional layer from odd neuron count.")
--  end
--  self.g_activation = nn.Sigmoid()
--  
--  self.i_count = self.max_seq_len * self.par_seq
--  self.in_weights = torch.Tensor(self.l_size, self.W_T_COUNT, self.i_size)
--  self.out_weights = torch.Tensor(self.l_size, self.W_T_COUNT, self.l_size)
--  self.acts = torch.Tensor(self.l_size, self.W_T_COUNT, self.i_count)
--  self.outputs = torch.Tensor(self.i_count, self.l_size)
--  self.cell_states = torch.Tensor(self.i_count, self.l_size)
--  self.b_weights = torch.Tensor(self.l_size, self.W_T_COUNT)
--  self.p_weights = torch.Tensor(self.l_size, self.P_W_COUNT)
--  transfer_to_gpu({self.inputs, self.in_weights, self.out_weights,
--                   self.acts, self.outputs, self.cell_states})
--end
--
--
--function Lstm:reset(stdv)
--   if stdv then
--      stdv = stdv * math.sqrt(3)
--   else
--      stdv = 0.1
--   end
--   
--  self.in_weights:uniform(-stdv, stdv)
--  self.out_weights:uniform(-stdv, stdv)
--  self.b_weights:uniform(-stdv, stdv)
--  self.p_weights:uniform(-stdv, stdv)
--  self.acts:zero()
--  self.outputs:zero()
--  self.cell_states:zero()
--  self.t_step = 0
--end
--
-- -- add input activations
--function Lstm:addInputActivations(input)
--  for i=1, self.l_size do
--    for y=1, self.W_T_COUNT do
--      self.acts[i][y] = input * self.in_weights[i][y]
--    end
--  end
--end
--
---- add output activations
--function Lstm:addOutputActivations()
--  for k=1,self.l_size do
--    for l=1,self.W_T_COUNT do
--      local outs = self.outputs[{{self.p_i_start, self.p_i_end},{}}]
--      self.acts[{k, l, {self.i_start, self.i_end}}]:addmv(outs, self.out_weights[k][l])
--    end  
--  end
--end
--
---- compute block activations
--function Lstm:computeBlocks()
--    for l=1, self.l_size do
--    local i_acts = self.acts[{l, self.I_GATE, {self.i_start, self.i_end}}]
--    local o_acts = self.acts[{l, self.O_GATE, {self.i_start, self.i_end}}]
--    local f_acts = self.acts[{l, self.F_GATE, {self.i_start, self.i_end}}]
--    local c_acts = self.acts[{l, self.C_STATE, {self.i_start, self.i_end}}]
--    local p_c_states
--    if self.t_step ~= 1 then
--      p_c_states = self.cell_states[{{self.p_i_start, self.p_i_end}, l}]
--    end
--    
--    i_acts:add(self.bias * self.b_weights[l][self.I_GATE])
--    o_acts:add(self.bias * self.b_weights[l][self.O_GATE])
--    f_acts:add(self.bias * self.b_weights[l][self.F_GATE])
--    c_acts:add(self.bias * self.b_weights[l][self.C_STATE])
--    
--    if self.t_step ~= 1 then
--      i_acts:add(p_c_states * self.p_weights[l][self.I_GATE])
--      f_acts:add(p_c_states * self.p_weights[l][self.F_GATE])
--    end
--      
--    c_acts:tanh()
--    i_acts = self.g_activation(i_acts)
--    f_acts = self.g_activation(f_acts)
--    
--    local a_c_states = c_acts * i_acts
--    if self.t_step ~= 1 then
--      a_c_states = a_c_states + f_acts*p_c_states
--    end
--    o_acts = self.g_activation(o_acts + a_c_states * self.p_weights[l][self.O_GATE])
--    
--    self.outputs[{{self.i_start,self.i_end},l}] = o_acts * torch.tanh(a_c_states)
--    self.cell_states[{{self.i_start, self.i_end}, l}] = a_c_states
--  end
--end
--
--function Lstm:updateOutput(input)
--  self:addInputActivations(input)
--  for t=1,self.max_seq_len do
--    self.t_step = t
--    self.i_start, self.i_end = self.par_seq*(t-1)+1, self.par_seq*t
--    self.p_i_start, self.p_i_end = (t-2)*self.par_seq+1, (t-1)*self.par_seq
--    if t ~= 1 then
--      self:addOutputActivations()
--    end
--    self:computeBlocks()
--  end
--  return self.outputs
--end
--
--local function main()
--  local par_seq = 50
--  local max_seq_len = 300
--  local i_count = par_seq * max_seq_len
--  local i_size = 100
--  local inputs = torch.randn(i_count, i_size)
--  
--  local params = {['bias']=1.0, ['l_size']=10, ['max_seq_len']=max_seq_len,
--                  ['input_size']=100, ['par_seq']=par_seq, ['i_size']=i_size, ['bidirectional']=true}
--  local layer = Lstm(params)
--  local outputs = layer(inputs)
--  collectgarbage()
--  print("end")
--end
--
--main()


--tester = torch.Tester()
--LstmTests = {}
--
--function LstmTests.testAddInputActivations()
--  local par_seq = 2
--  local max_seq_len = 3
--  local i_count = par_seq * max_seq_len
--  local i_size = 2
--  local inputs = torch.range(1,12):resize(par_seq * max_seq_len, i_size)
--  
--  local params = {['bias']=1.0, ['l_size']=3, ['max_seq_len']=max_seq_len,
--                  ['input_size']=100, ['par_seq']=par_seq, ['inputs']=inputs}
--  local layer = Lstm(params)
--  layer.in_weights = torch.range(1,24):resize(3,4,2)
--  layer:addInputActivations()
--  local result = {{{5,11,17,23,29,35},{11,25,39,53,67,81},{17,39,61,83,105,127},{23,53,83,113,143,173}},
--                  {{29,67,105,143,181,219},{35,81,127,173,219,265},{41,95,149,203,257,311},{47,109,171,233,295,357}},
--                  {{53,123,193,263,333,403},{59,137,215,293,371,449},{65,151,237,323,409,495},{71,165,259,353,447,541}}}
--  for i=1,layer.l_size do
--    for y=1,layer.W_T_COUNT do
--      for z=1, i_count do
--        tester:asserteq(layer.acts[i][y][z], result[i][y][z], "activations do not match on: " .. i .. y .. z)
--      end
--    end
--  end
--  
--  
--end
--
--function LstmTests.testAddOutputActivations()
--  local par_seq = 2
--  local max_seq_len = 3
--  local i_count = par_seq * max_seq_len
--  local i_size = 2
--  local inputs = torch.range(1,12):resize(par_seq * max_seq_len, i_size)
--  
--  local params = {['bias']=1.0, ['l_size']=2, ['max_seq_len']=max_seq_len,
--                  ['input_size']=100, ['par_seq']=par_seq, ['inputs']=inputs}
--  local layer = Lstm(params)
--  layer.out_weights = torch.range(1,16):resize(2,4,2)
--  layer.acts:fill(0)
--  layer.outputs[{{1,2},{}}] = torch.range(6):resize(2,3)
--  local outputs = layer:addOutputActivations(2)
--  local result = ({})
--end
--
--tester:add(LstmTests)
--tester:run()
