require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'ParallelTable'
require 'utils'


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
    instance:apply(function (m) m.bSizes = utils.getBatchSizes({3, 3}) end)
    local output = instance:forward(input)
    local err = instance:backward(input, output:clone())
end

for i = 1, #classNames do
    require(classNames[i])
end

classes = { nn.LinearScale, nn.LstmStep, nn.Lstm, nn.Blstm, nn.GruStep, nn.Gru, nn.Bgru, nn.RecLayer}

rnn = function(a, b, c) return nn.RecLayer(nn.Tanh, a, b, c) end
brnn = function(a, b, c) return nn.BiRecLayer(nn.Tanh, a, b, c) end


function testBatch(module)
    local iSize = 3
    local oSize = 6
    local hSize = 3
    local bSize = 2
    local b = module(iSize, oSize, hSize)
    local a = module(iSize, oSize, hSize)
    b:apply(function (m) m.bSizes = utils.getBatchSizes({3}) end)
    a:apply(function (m) m.bSizes = utils.getBatchSizes({3, 3}) end)
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

local batchModules = { nn.Bgru , nn.Blstm, nn.Gru, nn.Lstm, rnn, brnn}
local batchNames = { "Bgru", "Blstm", "Lstm", "Gru", "Rnn", "Brnn"}


function testBidirectional(bModule, uModule)
    local iSize = 2
    local oSize = 4
    local hSize = 3
    local wConst = 0.3
    local input = torch.ones(hSize, iSize)
    local bModel = bModule(iSize, oSize, hSize)
    local bx, bdx = bModel:getParameters()
    bx:fill(wConst)
    bModel:apply(function (m) m.bSizes = utils.getBatchSizes({3}) end)
    local uModel = uModule(iSize, oSize/2, hSize)
    local ux, udx = uModel:getParameters()
    ux:fill(wConst)
    uModel:apply(function (m) m.bSizes = utils.getBatchSizes({3}) end)

    local uOutput =uModel:forward(input)
    local bOutput = bModel:forward(input)

    tester:assertTensorEq(bOutput[{{}, {1, oSize/2}}], uOutput, "Outputs do not match.")

    local rUOutput = uOutput:clone()
    for i=1, rUOutput:size(1) do
        rUOutput[i] = uOutput[rUOutput:size(1) + 1 - i]
    end
    tester:assertTensorEq(bOutput[{{}, {oSize/2 + 1, oSize}}], rUOutput, "Outputs do not match.")

end

local biModules = { nn.Bgru, nn.Blstm, brnn}
local uModules = {nn.Gru, nn.Lstm, rnn}
local biNames = { "Gru", "Lstm", "Rnn"}

LstmTest = torch.TestSuite()


function testForwardBachward(module, e_output, e_error)
    local w_const = 0.3
    local history = 4
    local obj = module(1, 1, history)
    obj:apply(function (m) m.bSizes = utils.getBatchSizes({4}) end)
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


RnnTest = torch.TestSuite()

function RnnTest:RnnForwardBackward()
    local e_errors = {torch.Tensor({0.07335, 0.068707, 0.068108, 0.063992}),
                      torch.Tensor({0.2586, 0.2114, 0.1979, 0.1682}),
                      torch.Tensor({0.4251, 0.4170, 0.3900, 0.3000})}
    local e_outputs = {torch.Tensor({0.6457, 0.6886, 0.6914, 0.6916}),
                       torch.Tensor({0.5370, 0.6417, 0.6598, 0.6629}),
                       torch.Tensor({0.6, 0.78, 0.8340, 0.8502})}

    local aTypes = {nn.Sigmoid, nn.Tanh, nn.ReLU}
    for i=1, #aTypes do
        local func = function(iSize, oSize, history)
            return nn.RecLayer(aTypes[i], iSize, oSize, history)
        end
        testForwardBachward(func, e_outputs[i], e_errors[i])
    end
end

function testCtc()
    require 'warp_ctc'
    require 'CtcCriterion'
    local a = nn.CtcCriterion(3)
    require 'cutorch'
    local b = a:cuda()


    local acts = torch.Tensor({{1,2,3,4,5},{1,2,3,4,5},{-5,-4,-3,-2,-1},
                            {6,7,8,9,10},{6,7,8,9,10},{0,0,0,0,0},
                            {-5,4,3,-1,11},{11,12,13,14,15},{0,0,0,0,0}}):cuda()
    local labels = {{1}, {2,4}, {2}}
    local sizes = {3,3,1}
    local bSizes = {3,2,2}
    local grads = acts:clone():fill(0)
    local f = gpu_ctc(acts, grads, labels, sizes)
    local sum_f = utils.sumTable(f)
    local bActs = torch.Tensor({{1,2,3,4,5},{1,2,3,4,5},{-5,-4,-3,-2,-1},
                            {6,7,8,9,10},{6,7,8,9,10},
                            {-5,4,3,-1,11},{11,12,13,14,15}}):cuda()
    local err = b:forward(bActs, labels, sizes, bSizes)
    tester:asserteq(err, sum_f, "Ctc error does not fit.")
    b:backward(bActs, labels)
    -- sum should be the same
    local gSum = grads:sum(1)
    local gradInputSum = b.gradInput:sum(1)
    tester:assertTensorEq(gSum, gradInputSum, cond, "Ctc gradients do not fit.")
end


function testDifferentSeqLenghts(module)
    local iSize = 2
    local oSize = 4
    local aHist = 3
    local bhist = 5
    local cHist = 10
    local hSizes = {10, 5, 3}
    local mods = {}
    local inputs = {}
    local outputs = {}
    local mHist = utils.sumTable(hSizes)
    local mMod = module(iSize, oSize, mHist)
    mMod:apply(function (m) m.bSizes = utils.getBatchSizes(hSizes) end)
    local mInput = torch.ones(mHist, iSize)
    mMod:forward(mInput)
    local mX, mDx = mMod:getParameters()
    mDx:zero()
    local grads = {}
    for i=1, #hSizes do
        local mod = module(iSize, oSize, hSizes[i])
        mod:apply(function (m) m.bSizes = utils.getBatchSizes({hSizes[i]}) end)
        local x, dx  = mod:getParameters()
        dx:zero()
        table.insert(grads, dx)
        x:copy(mX)
        local inp = torch.ones(hSizes[i], iSize)
        mod:forward(inp)
        table.insert(mods, mod)
        table.insert(inputs, inp)
        table.insert(outputs, mod.output)
    end
    local eOutput = torch.Tensor(mHist, oSize)
    local index = 1
    for t=1, mHist do
        for i=1, #hSizes do
            if outputs[i]:size(1) >= t then
                eOutput[index]:copy(outputs[i][t])
                index = index + 1
            end
        end
    end
    tester:assertTensorEq(eOutput, mMod.output, cond, "Outputs are not correct")

    mMod:backward(mInput, mMod.output:clone():fill(1))
    local errors = {}
    for i=1, #hSizes do
        mods[i]:backward(inputs[i], mods[i].output:clone():fill(1))
        table.insert(errors, mods[i].gradInput)
    end
    local eErrors = torch.Tensor(mHist, iSize)
    index = 1
    for t=1, mHist do
        for i=1, #hSizes do
            if errors[i]:size(1) >= t then
                eErrors[index]:copy(errors[i][t])
                index = index + 1
            end
        end
    end
    tester:assertTensorEq(eErrors, mMod.gradInput, cond, "GradInputs are not correct")
    local eDx = grads[1]
    for i=2, #grads do
        eDx:add(grads[i])
    end
    tester:assertTensorEq(eDx,mDx, cond, "Gradiens are not correct")
end


for i = 1, #batchModules do
    local testFunction = function()
        testDifferentSeqLenghts(batchModules[i])
    end
    tester:add(testFunction, "TestDifferentSeqLenghts".. batchNames[i])
end

for i = 1, #classes do
    local testFunction = function()
        testClass(classes[i])
    end
    tester:add(testFunction, "BasicTest" .. classNames[i])
end

for i = 1, #batchModules do
    local testFunction = function()
        testBatch(batchModules[i])
    end
    tester:add(testFunction, "TestBatched".. batchNames[i])
end

for i = 1, #biModules do
    local testFunction = function()
        testBidirectional(biModules[i], uModules[i])
    end
    tester:add(testFunction, "TestBidirectional".. biNames[i])
end


tester:add(RnnTest)
tester:add(LstmTest)
tester:add(GruTest)
tester:add(testCtc, "CtcForwardBackward")


tester:run()
