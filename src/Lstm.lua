require 'torch'
require 'nn'
require 'Steps'
require 'LstmStep'


local Lstm = torch.class('nn.Lstm', 'nn.Sequential')

--TODO there is no support for non batch mode
--TODO resizing array should be optimized
--TODO biases arent correct in gates
-- TODO batch normalization -> previous linear layer should not have bias

function Lstm:__init(inputSize, layerSize, hist, bNorm)
    nn.Sequential.__init(self)
    --module for computing all input activations
    local aCount = 4 * layerSize
    self.layerSize = layerSize
    self.inputSize = inputSize
    -- set biases for all units in here -> temporary to one
    self.iActsModule = nn.Linear(inputSize, aCount)
    self.iActsModule.bias:fill(1)
    self:add(self.iActsModule)
    self.bNorm = bNorm or false
    if self.bNorm then
        self:add(nn.BatchNormalization(aCount))
    end
    local step = nn.LstmStep(layerSize, bNorm)
    self:add(nn.Steps(step, hist))
end


function Lstm:__tostring__()
    return torch.type(self) ..
            string.format('(%d -> %d, BatchNormalized=%s)', self.inputSize,
                self.layerSize,
                self.bNorm)
end

--function testBatched()
--  local iSize = 3
--  local oSize = 3
--  local hSize = 3
--  local bSize = 2
--  local b = nn.Lstm(iSize, oSize, hSize)
--  local a = nn.Lstm(iSize, oSize, hSize)
--  local x_b, _ = b:getParameters()
--  local x_a, _ = a:getParameters()
--  x_b:copy(x_a)
--  local inp = torch.ones(hSize, iSize)
--  local result = b:forward(inp):clone()
--  local errs = b:backward(inp, torch.range(1, oSize):repeatTensor(hSize, 1))
--  local e_errs = torch.Tensor(bSize * hSize, oSize)
--  local e_result = e_errs:clone()
--  local y = 1
--  for i=1, bSize*hSize, bSize do
--    e_result[i]:copy(result[y])
--    e_result[i+1]:copy(result[y])
--    e_errs[i]:copy(errs[y])
--    e_errs[i+1]:copy(errs[y])
--    y = y + 1
--  end
--  local batched_inp = torch.ones(bSize*hSize,3)
--  local a_result = a:forward(batched_inp)
--  local a_errs = a:backward(batched_inp, torch.range(1,3):repeatTensor(6,1))
--  if not torch.all(torch.eq(a_errs, e_errs)) then
--    print(a_errs)
--    print(e_errs)
--    print("Error: Backward pass does not match.")
--  end
--  if not torch.all(torch.eq(a_result, e_result)) then
--    print(a_result)
--    print(e_result)
--    print("Error: Forward pass does not match.")
--  end
--end
--
--function testCorrectForwardBackward()
--  local w_const = 0.3
--  local history = 3
--  local a = nn.Lstm(1,1,history)
--  local b = nn.LSTM(1,1)
--  local a_params = a:getParameters()
--  local b_params = b:getParameters()
--  a_params:fill(w_const)
--  b_params:fill(w_const)
--  local i_a = torch.ones(history,1)
--  local i_b = torch.ones(1)
--  local o_a = a:forward(i_a)
--  local o_b = b:forward(i_b):cat(b:forward(i_b)):cat(b:forward(i_b))
--  local e_a = a:backward(i_a, torch.ones(history,1))
--  local e_b = b:backward(i_b, torch.ones(1)):cat(b:backward(i_b, torch.ones(1))):cat(b:backward(i_b, torch.ones(1)))
--  print(e_b)
--  print(e_a)
--  if torch.ne(o_a, o_b):sum() ~= 0 then
--    print("Error: outputs are not consistent")
--    return 1
--  end
--  return 0
--end
--
--testCorrectForwardBackward()
--testBatched()

--local a = nn.Lstm(1,1,2)
--local b = nn.LSTM(1,1)
--local d_weight = 0.3
--local a_w, a_g = a:getParameters()
--local b_w, b_g = b:getParameters()
--b_w:fill(d_weight)
--a_w:fill(d_weight)
--local input = torch.ones(1)
--local a_output = a:forward(input:view(1,1))
--local b_output = b:forward(input)
--local err = torch.ones(1)
--local a_err = a:backward(input:view(1,1), err:view(1,1))
--local b_err = b:backward(input, err)
--print(b_err)
--print(a_err)

--checkBatchedForward()
--

--c = LinearScale.new(1)
--c:backward(torch.ones(1), torch.ones(1))
--a = nn.Sequential():add(Lstm.new(3,2,1)):add(Lstm.new(2,1,1))
--b = nn.Sequential():add(nn.LSTM(3,2)):add(nn.LSTM(2,1))
--
--a_params = a:getParameters()
--b_params = b:getParameters()
--
--a_params:fill(0.3)
--b_params:fill(0.3)
--
--print(a:forward(torch.ones(1,3)))
--print(b:forward(torch.ones(1,3)))
--print(a:backward(torch.ones(1,3), torch.ones(1,1)))
--print(b:backward(torch.ones(1,3), torch.ones(1,1)))

--a = Lstm.new(1, 2, 1)
--b = nn.LSTM(1,2)
--a:getParameters():fill(0.3)
--b:getParameters():fill(0.3)
--print(b:backward(torch.ones(1,1), torch.ones(1,1)))
--print(a:forward(torch.ones(1,1)))
--print(a:backward(torch.ones(1,1), torch.ones(1,1)))
--
--print(b:forward(torch.ones(1)))
--
--print(b:backward(torch.ones(1), torch.ones(2)))
--print(a:backward(torch.ones(1,1), torch.ones(1,2)))
--print(b:forward(torch.ones(1)))
--print(b:forward(torch.ones(1)))
--print(b:forward(torch.ones(1)))
--print(b:forward(torch.ones(1)))
--print(a.model)
--c = Lstm.new(1, 1, 3)
--b = nn.LSTM(1,2)
--print(b:getParameters():size())
--a_params = a:getParameters()
--print(a_params:size())
--b_params = c:getParameters()
--a_params:fill(0.2)
--b_params:fill(0.2)
--print(a:forward(torch.ones(1,1)))
--print(c:forward(torch.ones(1,1)))
--
--print(a_params:size())
--print(b:getParameters():size())
--print(c:forward(torch.ones(1,1)))
--print(b:forward(torch.ones(1)))
--print(b:forward(torch.ones(1)))
--print(b:forward(torch.ones(1)))


--print(a.model)
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
--- -print("acts")
---- print(a.a_i_acts_module.output)
-- print(a:getCellStates(a.modules[2]))
-- print(a:getCellStates(a.modules[3]))
-- print(a.model:get(8).output[1])
-- print(a.model:get(8).output[2])
-- print(a.model:get(8).output[3])

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

--eof
