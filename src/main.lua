require 'torch' 
require 'cutorch'
require 'nn'


-- function for transfering data to gpu memory
local function transfer_to_gpu(data)
  for _, i in ipairs(data) do
    i:cuda()
  end
end

local LstmLayer, parent = torch.class('LstmLayer', 'nn.Module')
--local BiLstmLayer, parent = torch.class('BiLstmLayer', 'nn.Module')
--
--function BiLstmLayer:__init(params)
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
--  self.fw_layer = LstmLayer(params)
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
--  self.bw_layer = LstmLayer(bw_params)
--end
--
--function BiLstmLayer:forward()
--  local f_outputs = self.fw_layer:forward()
--  local b_outputs = self.bw_layer:forward()
--  local middle = self.outputs:size(2) / 2
--  for i=1,self.outputs:size(1) do
--    self.outputs[{{i},{1,middle}}] = f_outputs[i]
--    self.outputs[{{i}, {middle+1, -1}}] = b_outputs[i]
--  end
--  return self.outputs
--end

function LstmLayer:__init(params)
  parent.__init(self)
  for item, value in pairs(params) do
    self[item] = value
  end
  
  if self.bidirectional and self.l_size % 2 ~= 0 then
    error("Cannot consturct bidirectional layer from odd neuron count.")
  end
  
  self.I_GATE = 1
  self.F_GATE = 2
  self.O_GATE = 3
  self.C_STATE =4
  
  self.W_T_COUNT = 4
  self.P_W_COUNT = 3
  
  self.g_activation = nn.Sigmoid()
  
  self.i_count = self.max_seq_len * self.par_seq
  self.in_weights = torch.Tensor(self.l_size, self.W_T_COUNT, self.i_size)
  self.out_weights = torch.Tensor(self.l_size, self.W_T_COUNT, self.l_size)
  self.acts = torch.Tensor(self.l_size, self.W_T_COUNT, self.i_count)
  self.outputs = torch.Tensor(self.i_count, self.l_size)
  self.cell_states = torch.Tensor(self.i_count, self.l_size)
  self.b_weights = torch.Tensor(self.l_size, self.W_T_COUNT)
  self.p_weights = torch.Tensor(self.l_size, self.P_W_COUNT)
  transfer_to_gpu({self.inputs, self.in_weights, self.out_weights,
                   self.acts, self.outputs, self.cell_states})
end


function LstmLayer:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 0.1
   end
   
  self.in_weights:uniform(-stdv, stdv)
  self.out_weights:uniform(-stdv, stdv)
  self.b_weights:uniform(-stdv, stdv)
  self.p_weights:uniform(-stdv, stdv)
  self.acts:zero()
  self.outputs:zero()
  self.cell_states:zero()
  self.t_step = 0
end

 -- add input activations
function LstmLayer:addInputActivations(input)
  for i=1, self.l_size do
    for y=1, self.W_T_COUNT do
      self.acts[i][y] = input * self.in_weights[i][y]
    end
  end
end

-- add output activations
function LstmLayer:addOutputActivations()
  for k=1,self.l_size do
    for l=1,self.W_T_COUNT do
      local outs = self.outputs[{{self.p_i_start, self.p_i_end},{}}]
      self.acts[{k, l, {self.i_start, self.i_end}}]:addmv(outs, self.out_weights[k][l])
    end  
  end
end

-- compute block activations
function LstmLayer:computeBlocks()
    for l=1, self.l_size do
    local i_acts = self.acts[{l, self.I_GATE, {self.i_start, self.i_end}}]
    local o_acts = self.acts[{l, self.O_GATE, {self.i_start, self.i_end}}]
    local f_acts = self.acts[{l, self.F_GATE, {self.i_start, self.i_end}}]
    local c_acts = self.acts[{l, self.C_STATE, {self.i_start, self.i_end}}]
    local p_c_states
    if self.t_step ~= 1 then
      p_c_states = self.cell_states[{{self.p_i_start, self.p_i_end}, l}]
    end
    
    i_acts:add(self.bias * self.b_weights[l][self.I_GATE])
    o_acts:add(self.bias * self.b_weights[l][self.O_GATE])
    f_acts:add(self.bias * self.b_weights[l][self.F_GATE])
    c_acts:add(self.bias * self.b_weights[l][self.C_STATE])
    
    if self.t_step ~= 1 then
      i_acts:add(p_c_states * self.p_weights[l][self.I_GATE])
      f_acts:add(p_c_states * self.p_weights[l][self.F_GATE])
    end
      
    c_acts:tanh()
    i_acts = self.g_activation(i_acts)
    f_acts = self.g_activation(f_acts)
    
    local a_c_states = c_acts * i_acts
    if self.t_step ~= 1 then
      a_c_states = a_c_states + f_acts*p_c_states
    end
    o_acts = self.g_activation(o_acts + a_c_states * self.p_weights[l][self.O_GATE])
    
    self.outputs[{{self.i_start,self.i_end},l}] = o_acts * torch.tanh(a_c_states)
    self.cell_states[{{self.i_start, self.i_end}, l}] = a_c_states
  end
end

function LstmLayer:updateOutput(input)
  self:addInputActivations(input)
  for t=1,self.max_seq_len do
    self.t_step = t
    self.i_start, self.i_end = self.par_seq*(t-1)+1, self.par_seq*t
    self.p_i_start, self.p_i_end = (t-2)*self.par_seq+1, (t-1)*self.par_seq
    if t ~= 1 then
      self:addOutputActivations()
    end
    self:computeBlocks()
  end
  return self.outputs
end

local function main()
  local par_seq = 50
  local max_seq_len = 300
  local i_count = par_seq * max_seq_len
  local i_size = 100
  local inputs = torch.randn(i_count, i_size)
  
  local params = {['bias']=1.0, ['l_size']=10, ['max_seq_len']=max_seq_len,
                  ['input_size']=100, ['par_seq']=par_seq, ['i_size']=i_size, ['bidirectional']=true}
  local layer = LstmLayer(params)
  local outputs = layer(inputs)
  collectgarbage()
  print("end")
end

main()


--tester = torch.Tester()
--LstmLayerTests = {}
--
--function LstmLayerTests.testAddInputActivations()
--  local par_seq = 2
--  local max_seq_len = 3
--  local i_count = par_seq * max_seq_len
--  local i_size = 2
--  local inputs = torch.range(1,12):resize(par_seq * max_seq_len, i_size)
--  
--  local params = {['bias']=1.0, ['l_size']=3, ['max_seq_len']=max_seq_len,
--                  ['input_size']=100, ['par_seq']=par_seq, ['inputs']=inputs}
--  local layer = LstmLayer(params)
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
--function LstmLayerTests.testAddOutputActivations()
--  local par_seq = 2
--  local max_seq_len = 3
--  local i_count = par_seq * max_seq_len
--  local i_size = 2
--  local inputs = torch.range(1,12):resize(par_seq * max_seq_len, i_size)
--  
--  local params = {['bias']=1.0, ['l_size']=2, ['max_seq_len']=max_seq_len,
--                  ['input_size']=100, ['par_seq']=par_seq, ['inputs']=inputs}
--  local layer = LstmLayer(params)
--  layer.out_weights = torch.range(1,16):resize(2,4,2)
--  layer.acts:fill(0)
--  layer.outputs[{{1,2},{}}] = torch.range(6):resize(2,3)
--  local outputs = layer:addOutputActivations(2)
--  local result = ({})
--end
--
--tester:add(LstmLayerTests)
--tester:run()
