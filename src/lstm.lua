require 'torch'
require 'nn'

local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
   nn.Module.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function LinearNoBias:reset(stdv)
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

function LinearNoBias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:mv(self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end


local Lstm, parent = torch.class('Lstm', 'nn.Container')

--TODO there is no support for non batch mode
--TODO resizing array should be optimized
--TODO biases arent correct in gates

local LstmStep, _ = torch.class('LstmStep', 'nn.Sequential')

local CELL_MODULE_INDEX = 8

-- It is necessary to supply gradoutput and cellGrads from next timestemp
function LstmStep:backward(input, gradOutput, pGradOutput, pCellGrad, scale)
   gradOutput:add(pGradOutput)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if i == CELL_MODULE_INDEX then -- we should add previous cell errors
        currentGradOutput[1]:add(pCellGrad)
      end
      currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Lstm:__init(inputSize, layerSize, hist, b_norm)
  parent.__init(self)
  self.batch_size = 0
  self.b_norm =  b_norm or false
  self.scale = 1 -- 1 / hist
  self.inputSize = inputSize
  self.history_size = hist -- history size
  self.layerSize = layerSize
  self.a_i_acts = nil -- input activations
  self.z_tensor = torch.zeros(1, self.layerSize) -- zero input
  self.g_output = torch.Tensor() -- temporary gradients
  --module for computing all input activations
  self.a_count = 4 * layerSize
  local p_count = 2 * layerSize
  -- set biases for all units in here -> temporary to one
  self.a_i_acts_module = nn.Sequential()
  self.a_i_acts_module:add(nn.Linear(inputSize, self.a_count))
--  if self.b_norm then
--    self.a_i_acts_module:add(nn.BatchNormalization(self.a_count))
--  end
  table.insert(self.modules, self.a_i_acts_module)
  --module for computing one mini batch
  self.model = LstmStep.new()
  local i_acts = nn.Identity()
  -- all output activations
  local o_acts = nn.Sequential():add(LinearNoBias.new(layerSize, self.a_count))
--  if self.b_norm then
--    o_acts:add(nn.BatchNormalization(self.a_count))
--  end
  -- forget and input peepholes cell acts
  local fg_peep = nn.Sequential():add(LinearNoBias.new(layerSize, p_count))
--  if self.b_norm then
--    fg_peep:add(nn.BatchNormalization(p_count))
--  end
  local c_acts = nn.ConcatTable():add(fg_peep):add(nn.Identity())

  -- container for summed input and output activations
  -- that is divided in half
  local io_acts = nn.Sequential()
  io_acts:add(nn.ParallelTable():add(i_acts):add(o_acts))
  io_acts:add(nn.CAddTable()):add(nn.Reshape(2, 2 * layerSize)):add(nn.SplitTable(1,2))
  -- sum half of the activations with peepholes
  self.model:add(nn.ParallelTable():add(io_acts):add(c_acts))
  self.model:add(nn.FlattenTable())
  -- output of the model at this stage is <c_states + o_acts, i_acts + f_acts, peepholes acts, cell states>
  -- input and forget gate activation
  local gates = nn.ConcatTable()
  gates:add(nn.Sequential():add(nn.NarrowTable(2,2)):add(nn.CAddTable()):add(nn.Sigmoid()):add(nn.Reshape(2, layerSize)):add(nn.SplitTable(1,2)))
  gates:add(nn.Sequential():add(nn.SelectTable(4)))
  --  -- divide rest activations between cell state and output gate
  gates:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Reshape(2, layerSize)):add(nn.SplitTable(1,2)))
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
  if self.b_norm then
    cell_acts:add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()):add(nn.BatchNormalization(self.layerSize)))
  else
    cell_acts:add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
  end
  cell_acts:add(nn.Sequential():add(nn.SelectTable(3)))
  self.model:add(cell_acts)
  -- output of the model at this stage is <c_acts, o_acts>
  -- scale by peephole from the cell state to output gate and apply sigmoid to output gate,
  -- also apply squashing function to the cell states
  cell_acts = nn.ConcatTable()
--  if self.b_norm then
--    cell_acts:add(nn.Sequential():add(nn.SelectTable(1)):add(LinearNoBias.new(layerSize, layerSize)):add(nn.BatchNormalization(layerSize)))
--  else
  cell_acts:add(nn.Sequential():add(nn.SelectTable(1)):add(LinearNoBias.new(layerSize, layerSize)))
--  end
  cell_acts:add(nn.SelectTable(2))
  cell_acts:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Tanh()))
  self.model:add(cell_acts)
  -- output of the model at this stage is <output_gate peephole act, o_acts, cell_acts>
  -- finalize the o_acts and apply sigmoid
  local cell_acts = nn.ConcatTable():add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()):add(nn.Sigmoid()))
  -- just forward cell acts
  cell_acts:add(nn.SelectTable(3))
  -- result is <output>
  self.model:add(cell_acts)
  self.model:add(nn.CMulTable())
  -- copies of 'step' module
  table.insert(self.modules, self.model)
  for i=2, hist do
    table.insert(self.modules, self.model:clone('weight','bias','gradWeight','gradBias'))
  end
end

function Lstm:updateGradInput(input, gradOutput)
  self.g_output:resize(self.batch_size * self.history_size, self.a_count)
  local size = gradOutput:size(1)
  local interval = {size - self.batch_size + 1, size} 
  local z_tensor = self.z_tensor:repeatTensor(self.batch_size, 1)
  local inp = {{self.a_i_acts[{interval, {}}], z_tensor}, z_tensor}
  local tmp_g = gradOutput[{interval, {}}]
  -- first do propagation from the last module
  local l_step = self.modules[#self.modules]
  l_step:backward(inp, tmp_g, z_tensor, z_tensor, self.scale)
  self.g_output[{interval, {}}]:copy(l_step.gradInput[1][1]) -- error for the next layer
  local p_o_grad = z_tensor:clone()
  local p_c_grad = z_tensor:clone()
  local counter = 2
  for i=1, #self.modules - 2 do
    local c = #self.modules - i
    local step = self.modules[c]
    local p_step = self.modules[c+1]
    interval = {size - counter * self.batch_size +1, size - (counter -1) * self.batch_size}
    local inp = {{self.a_i_acts[{interval,{}}], p_step.output}, self:getCellStates(p_step)}
    -- propagate error from previous time step
    step:backward(inp, gradOutput[{interval, {}}], p_step.gradInput[1][2], p_step.gradInput[2], self.scale)
    self.g_output[{interval, {}}]:copy(step.gradInput[1][1])
--    -- acumulate error
--    self.g_output[{interval, {}}]:add(p_step.gradInput[1][1])
    counter = counter + 1
  end
  
  self.a_i_acts_module:backward(input, self.g_output, self.step)
  self.gradInput:resizeAs(self.a_i_acts_module.gradInput)
  self.gradInput:copy(self.a_i_acts_module.gradInput)
  return self.gradInput
end

function Lstm:updateOutput(input)
  -- TODO this resizing might be handled better
--  if not self.train then -- threat input as one long sequence
--    self.output:resize(input:size(1), self.layerSize)
--    self.a_i_acts = self.a_i_acts_module:forward(input)
--    self.model:forward({{self.a_i_acts[{{1}}], self.z_tensor[{{1}}]}, self.z_tensor[{{1}}]})
--    self.output[1] = self.model.output
--    for i=2, input:size(1) do
--      self.model:forward({{self.a_i_acts[{{i}}], self.model.output}, self:getCellStates(self.model)})
--      self.output[i] = self.model.output
--    end
--    return self.output
--  else -- training mode
    if not self.train and input:size(1) < self.history_size then
      -- there can be sequence that is shorter than required history
      self.history_backup = self.history_size
      self.history_size = input:size(1)        
    end 
    self.batch_size = input:size(1) / self.history_size
    self.output:resize(self.history_size * self.batch_size, self.layerSize)
    self.a_i_acts = self.a_i_acts_module:forward(input)
    local z_tensor = self.z_tensor:repeatTensor(self.batch_size, 1)
    -- do first step manually, set previous output and previous cell state to zeros
    self.model:forward({{self.a_i_acts[{{1, self.batch_size}}], z_tensor}, z_tensor})
    self.output[{{1, self.batch_size}}]:copy(self.model.output)
    for i= 2, input:size(1)/self.batch_size do
      local p_step = self.modules[i]
      local step = self.modules[i+1]
      local interval = {(i-1)*self.batch_size + 1, i*self.batch_size}
      local t_i_acts = self.a_i_acts[{interval}]
      step:forward({{t_i_acts, p_step.output}, self:getCellStates(p_step)})
      self.output[{interval,{}}]:copy(step.output)
    end
--  end
  if not self.train and self.history_backup then
    self.history_size = self.history_backup
    self.history_backup = nil
  end
  return self.output
end

function Lstm:getCellStates(model)
  -- gathers cell states from particular module
  return model:get(7).output[1]
end

function Lstm:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d, BatchNormalized=%s)', self.inputSize, self.layerSize, self.b_norm)
end

function checkBatchedForward()
  local a = Lstm.new(3,3,3)
  local inp = torch.ones(3,3)
  local result = a:forward(inp):clone()
  local e_result = torch.Tensor(6,3)
  
  local y = 1
  for i=1,6,2 do
    e_result[i]:copy(result[y])
    e_result[i+1]:copy(result[y])
    y = y + 1
  end
  local batched_inp = torch.ones(6,3)
  local a_result = a:forward(batched_inp)
  
  if not torch.all(torch.eq(a_result, e_result)) then
    perror("results do not match")
    return 1
  end
end

--checkBatchedForward()
--
--a = Lstm.new(1, 1, 3)
--
--f, c = a:getParameters()
--c:zero()
--print(c)
--print(a:parameters())
--a:training()
--x,f = a:getParameters()

--print(x:size(1),f:size(1))
--f:zero()
--print(f)
--x:fill(0.25)
--a.a_i_acts_module.bias:fill(0)
--inp = torch.Tensor(6,1):fill(1)
--print(a:forward(inp))
--a:backward(inp, inp)
--print(f)
--print(a.a_i_acts_module.gradWeight)
--print(f)
--print(a.gradWeight)

--a:backward()
----print("acts")
----print(a.a_i_acts_module.output)
--print(a:getCellStates(a.modules[2]))
--print(a:getCellStates(a.modules[3]))
--print(a.model:get(8).output[1])
--print(a.model:get(8).output[2])
--print(a.model:get(8).output[3])

--print(a:getParameters())
--a.a_i_acts_module:getparameter
--c,_ = a:getParameters()
--print(c)
--print(a.a_i_acts_module)

--print(a.model)
--inp = torch.randn(16*50,10)
--
--a:training() --evaluate()
--_,b = a:getParameters()
--output = a(inp)
--err = torch.randn(16*50, 20)
--a:backward(inp, err)
--a:backward(inp, err)
--a:backward(inp, err)
--a:backward(inp, err)

return Lstm



