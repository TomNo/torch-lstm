require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'ParallelTable'
require 'MaskedCECriterion'

torch.class("nn.RegularTanh", "nn.Tanh")
torch.class("nn.RegularSigmoid", "nn.Sigmoid")

--TODO generalize - to much code duplication :D!!!
-- TODO bidirectional and simple rnn

tester = torch.Tester()


classNames = { "LinearScale", "LstmStep", "Lstm", "Blstm", "GruStep", "Gru", "Bgru", "RecLayer" }


cond = 1e-4


--[[
-- Test basic functionality backward/forward of every class
 ]]
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



--[[
-- Checks that batched output corresponds to classic forward
 ]]
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


--[[
-- Checks that bidirectional outputs are the same in both directions
 ]]
function testBidirectional(bModule, uModule)
    local iSize = 2
    local oSize = 4
    local hSize = 3
    local wConst = 0.3
    local input = torch.ones(hSize, iSize)
    local bModel = bModule(iSize, oSize, hSize)
    local bx, bdx = bModel:getParameters()
    bx:fill(wConst)
    bModel:apply(function (m) m.sizes = {3} end)
    local uModel = uModule(iSize, oSize/2, hSize)
    local ux, udx = uModel:getParameters()
    ux:fill(wConst)
    uModel:apply(function (m) m.sizes = {3} end)

    local uOutput =uModel:forward(input)
    local bOutput = bModel:forward(input)

    tester:assertTensorEq(bOutput[{{}, {1, oSize/2}}], uOutput, "Outputs do not match.")

    local rUOutput = uOutput:clone()
    for i=1, rUOutput:size(1) do
        rUOutput[i] = uOutput[rUOutput:size(1) + 1 - i]
    end
    tester:assertTensorEq(bOutput[{{}, {oSize/2 + 1, oSize}}], rUOutput, "Outputs do not match.")

end


local biModules = { nn.Bgru, nn.Blstm}
local uModules = {nn.Gru, nn.Lstm}
local biNames = { "Gru", "Lstm"}

for i = 1, #biModules do
    local testFunction = function()
        testBidirectional(biModules[i], uModules[i])
    end
    tester:add(testFunction, "TestBidirectional".. biNames[i])
end


--[[
-- Function for checking correct forward/backward outputs
 ]]
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


LstmTest = torch.TestSuite()

--[[
-- Test correct lstm forward/backward
 ]]
function LstmTest.LstmForwardBackward()
    local e_output = torch.Tensor({0.2231, 0.3946, 0.5189, 0.6064})
    local e_error = torch.Tensor({0.3165, 0.2701, 0.1930, 0.1125})
    testForwardBachward(nn.Lstm, e_output, e_error)
end


GruTest = torch.TestSuite()

--[[
-- Test correct gru forward/backward
 ]]
function GruTest:GruForwardBackward()
    local e_error = torch.Tensor({0.096448, 0.099792, 0.085345, 0.052972})
    local e_output = torch.Tensor({0.1903, 0.3176,0.4052,0.4665})
    testForwardBachward(nn.Gru, e_output, e_error)
end


--[[
-- Test that warp_ctc wrapping works as expected
 ]]
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


--[[
-- Test that cross entropy with masked gradInput works
 ]]
function testMaskedCE()
    local iSize = 3
    local iCount = 5
    local ce = nn.CrossEntropyCriterion()
    local m_ce = nn.MaskedCECriterion()
    local ce_input = torch.ones(iCount, iSize)
    local m_ce_input = torch.ones(iCount * 2,iSize)
    local ce_labels = torch.ones(iCount)
    local m_ce_labels = torch.ones(iCount * 2)
    m_ce_labels[{{1, iCount}}]:fill(0)
    local ce_err = ce:forward(ce_input, ce_labels)
    local m_ce_err = m_ce:forward(m_ce_input, m_ce_labels)
    tester:asserteq(ce_err, m_ce_err, "Masked CE error does not fit.")
    ce:backward(ce_input, ce_labels)
    m_ce:backward(m_ce_input, m_ce_labels)
    tester:assertTensorEq(ce.gradInput, m_ce.gradInput[{{iCount + 1, 2*iCount}}], cond, "Masked CE gradient does not fit")
    tester:assertTensorEq(torch.zeros(iCount, iSize),  m_ce.gradInput[{{1, iCount}}], "Masked CE gradient does not fit")
end


--[[
-- Test that results is the same when having different length of the time steps
 ]]
function testSteps(module)
    local bHist = 3
    local aHist = 5
    local iSize = 1
    local oSize = 2
    local aModule = module(iSize, oSize, aHist)
    local a_x, a_dx = aModule:getParameters()
    a_dx:zero()
    local bModule = module(iSize, oSize, bHist)
    local b_x, b_dx = bModule:getParameters()
    b_dx:zero()
    b_x:copy(a_x)
    local input = torch.zeros(aHist, iSize)
    input[{{1, bHist}}]:fill(1)
    local sFunc = function(m) m.sizes = {bHist} end
    aModule:apply(sFunc)
    bModule:apply(sFunc)
    aModule:forward(input)
    bModule:forward(input[{{1, bHist}}])
    tester:assertTensorEq(aModule.output[{{1, bHist}}], bModule.output, cond, "Outputs do not fit.")
    tester:assertTensorEq(aModule.output[{{bHist + 1, aHist}}], torch.zeros(aHist - bHist, oSize))
    local err_a = aModule.output:clone():fill(1)
    err_a[{{bHist+1, aHist}}]:fill(0)
    local err_b = bModule.output:clone():fill(1)
    aModule:backward(input, err_a)
    bModule:backward(input[{{1, bHist}}], err_b)
    tester:assertTensorEq(aModule.gradInput[{{1, bHist}}], bModule.gradInput, cond, "GradInputs are not equal.")
    tester:assertTensorEq(aModule.gradInput[{{bHist + 1, aHist }}], torch.zeros(aHist - bHist, iSize), "GradInputs are not equal.")
    tester:assertTensorEq(a_dx, b_dx, cond, "Gradiends are not equal.")
end

for i=1, #batchModules do
    local tmp =  function ()
        testSteps(batchModules[i])
    end
    tester:add(tmp,"TestSeqSizes" .. batchNames[i])
end



tester:add(LstmTest)
tester:add(GruTest)
tester:add(testCtc, "CtcForwardBackward")
tester:add(testMaskedCE, "MaskedCECriterionForwardBackward")

tester:run()
