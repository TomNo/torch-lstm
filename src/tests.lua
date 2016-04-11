require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'ParallelTable'

torch.class("nn.RegularTanh", "nn.Tanh")

--TODO generalize - to much code duplication :D!!!
-- TODO bidirectional and simple rnn

tester = torch.Tester()


classNames = { "LinearScale", "LstmStep", "Lstm", "Blstm", "GruStep", "Gru", "Bgru", "RecLayer" }


cond = 1e-4

function testClass(class)
    local lSize = 6
    local iSize = 6
    local history = 3
    local bSize = 2
    local bNorm = false
    local instance
    if class.__typename == "nn.Lstm" or class.__typename == "nn.Blstm" or
            class.__typename == 'nn.Gru' or class.__typename == 'nn.Bgru' then
        instance = class.new(iSize, lSize, history, bNorm)
    elseif class.__typename == "nn.RecLayer" then
        instance = class.new(nn.Tanh, iSize, lSize, history, bNorm)
    else
        instance = class.new(lSize, bNorm, history)
    end
    instance = instance:cuda()
    local input
    if class.__typename == "nn.LstmStep" then
        input = torch.ones(bSize * history, lSize * 4):cuda()
    elseif class.__typename == "nn.GruStep" then
        input = torch.ones(bSize * history, lSize * 3):cuda()
    elseif class.__typename == "nn.LinearScale" then
        local tmp = torch.ones(bSize * history, lSize):cuda()
        input = {tmp, tmp}
    else
        input = torch.ones(bSize * history, lSize):cuda()
    end
    instance:apply(function (m) m.sizes = {3, 3} end)
    local output = instance:forward(input)
    local err = instance:backward(input, output:clone())
end

for i = 1, #classNames do
    require(classNames[i])
end

classes = { nn.LinearScale, nn.LstmStep, nn.Lstm, nn.Blstm, nn.GruStep, nn.Gru, nn.Bgru, nn.RecLayer}

for i = 1, #classes do
    local testFunction = function()
        testClass(classes[i])
    end
    tester:add(testFunction, "BasicTest" .. classNames[i])
end


function testBatch(module)
    local iSize = 3
    local oSize = 6
    local hSize = 3
    local bSize = 2
    local b = module(iSize, oSize, hSize)
    local a = module(iSize, oSize, hSize)
    b:apply(function (m) m.sizes = {3, 3} end)
    a:apply(function (m) m.sizes = {3, 3} end)
    local x_b, _ = b:getParameters()
    local x_a, _ = a:getParameters()
    x_b:copy(x_a)
    local inp = torch.ones(hSize, iSize)
    local result = b:forward(inp):clone()
    local errs = b:backward(inp, torch.range(1, oSize):repeatTensor(hSize, 1))
    local e_errs = torch.Tensor(bSize * hSize, iSize)
    local e_result = torch.Tensor(hSize * bSize, oSize)
    local y = 1
    for i = 1, bSize * hSize, bSize do
        e_result[i]:copy(result[y])
        e_result[i + 1]:copy(result[y])
        e_errs[i]:copy(errs[y])
        e_errs[i + 1]:copy(errs[y])
        y = y + 1
    end
    local batched_inp = torch.ones(bSize * hSize, iSize)
    local a_result = a:forward(batched_inp)
    local a_errs = a:backward(batched_inp, torch.range(1, oSize):repeatTensor(oSize, 1))
    tester:assertTensorEq(a_result, e_result, cond, "Forward pass do not match.")
    tester:assertTensorEq(a_errs, e_errs, cond, "Backward pass does not match.")
end

local batchModules = { nn.Bgru, nn.Blstm, nn.Gru, nn.Lstm}
local batchNames = { "Bgru", "Blstm", "Lstm", "Gru", "Bgru", "Blstm"}

for i = 1, #batchModules do
    local testFunction = function()
        testBatch(batchModules[i])
    end
    tester:add(testFunction, "TestBatched".. batchNames[i])
end


--function testBidirectional(bModule, uModule)
--    local iSize = 2
--    local oSize = 4
--    local hSize = 3
--    local input = torch.ones(hSize, iSize)
--    local bModel = bModule(iSize, oSize, hSize)
--    print(bModel)
--    local uModel = uModule(iSize, oSize/2, hSize)
--    print(input)
--    local uOutput =uModule:forward(input)
--    print(input)
--    local bOutput = bModule:forward(input)
--    tester:assertTensorEq(bOutput[{{1, oSize/2}}], uOutput, "Outputs do not match.")
--    tester:assertTensorEq(bOutput[{{oSize/2 + 1, oSize}}], uOutput, "Outputs do not match.")
--
--end
--
--
--local biModules = { nn.Bgru, nn.Blstm}
--local uModules = {nn.Gru, nn.Lstm}
--local biNames = { "Gru", "Lstm"}
--
--for i = 1, #biModules do
--    local testFunction = function()
--        testBidirectional(biModules[i], uModules[i])
--    end
--    tester:add(testFunction, "TestBidirectional".. biNames[i])
--end

LstmTest = torch.TestSuite()


function testForwardBachward(module, e_output, e_error)
    local w_const = 0.3
    local history = 4
    local obj = module(1, 1, history)
    obj:apply(function (m) m.sizes = {4} end)
    local params = obj:getParameters()
    params:fill(w_const)
    local input = torch.ones(history, 1)
    local output = obj:forward(input)
    local error = obj:backward(input, torch.ones(history, 1)):squeeze()
    tester:assertTensorEq(output:squeeze(), e_output, cond, "Outputs do not match.")
    tester:assertTensorEq(error, e_error, cond, "Errors do not match.")
end

function LstmTest.LstmForwardBackward()
    local e_output = torch.Tensor({0.2231, 0.3946, 0.5189, 0.6064})
    local e_error = torch.Tensor({0.3165, 0.2701, 0.1930, 0.1125})
    testForwardBachward(nn.Lstm, e_output, e_error)
end


GruTest = torch.TestSuite()

function GruTest:GruForwardBackward()
    local e_error = torch.Tensor({0.096448, 0.099792, 0.085345, 0.052972})
    local e_output = torch.Tensor({0.1903, 0.3176,0.4052,0.4665})
    testForwardBachward(nn.Gru, e_output, e_error)
end

function testCtc()
    require 'warp_ctc'
    require 'CtcCriterion'
    local a = nn.CtcCriterion(3)
    require 'cutorch'
    local b = a:cuda()


    local acts = torch.Tensor({{1,2,3,4,5},{1,2,3,4,5},{-5,-4,-3,-2,-1},
                            {6,7,8,9,10},{6,7,8,9,10},{-10,-9,-8,-7,-6},
                            {-5,4,3,-1,11},{11,12,13,14,15},{-15,-14,-13,-12,-11}}):cuda()
    local labels = {{1}, {2,4}, {2,3}}
    local sizes = {3,3,3}
    local grads = acts:clone():fill(0)
    local f = gpu_ctc(acts, grads, labels, sizes)
    local sum_f = sumTable(f)

    local cLabels = torch.Tensor({1,2,2,1,2,3,1,4,3})
    local err = b:forward(acts, cLabels)

    tester:asserteq(err, sum_f, "Ctc error does not fit.")
    tester:assertTensorEq(grads, b:backward(), cond, "Ctc gradients does not fit.")
end


tester:add(LstmTest)
tester:add(GruTest)
tester:add(testCtc, "CtcForwardBackward")


tester:run()
