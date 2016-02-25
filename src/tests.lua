require 'torch'
require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'


--TODO generalize - to much code duplication :D

tester = torch.Tester()


classNames = { "LinearScale", "LstmStep", "Lstm", "Blstm", "GruStep", "Gru", "Bgru" }


function testClass(class)
    local lSize = 6
    local iSize = 6
    local history = 3
    local bSize = 2
    local bNorm = false
    local instance
    if class.__typename == "nn.Lstm" or class.__typename == "nn.Blstm" or
            class.__typename == 'nn.Gru' or class.__typename == 'nn.Bgru' then
        instance = class.new(iSize, lSize, history, bNorm):cuda()
    else
        instance = class.new(lSize, bNorm, history):cuda()
    end
    local input
    if class.__typename == "nn.LstmStep" then
        input = torch.ones(bSize * history, lSize * 4):cuda()
    elseif class.__typename == "nn.GruStep" then
        input = torch.ones(bSize * history, lSize * 3):cuda()
    else
        input = torch.ones(bSize * history, lSize):cuda()
    end

    local output = instance:forward(input)
    local err = instance:backward(input, output:clone())
end

for i = 1, #classNames do
    require(classNames[i])
end

classes = { nn.LinearScale, nn.LstmStep, nn.Lstm, nn.Blstm, nn.GruStep, nn.Gru, nn.Bgru }

for i = 1, #classes do
    local testFunction = function()
        testClass(classes[i])
    end
    tester:add(testFunction, "BasicTest" .. classNames[i])
end

LstmTest = {}


function LstmTest:testBatched()
    local iSize = 3
    local oSize = 3
    local hSize = 3
    local bSize = 2
    local b = nn.Lstm(iSize, oSize, hSize)
    local a = nn.Lstm(iSize, oSize, hSize)
    local x_b, _ = b:getParameters()
    local x_a, _ = a:getParameters()
    x_b:copy(x_a)
    local inp = torch.ones(hSize, iSize)
    local result = b:forward(inp):clone()
    local errs = b:backward(inp, torch.range(1, oSize):repeatTensor(hSize, 1))
    local e_errs = torch.Tensor(bSize * hSize, oSize)
    local e_result = e_errs:clone()
    local y = 1
    for i = 1, bSize * hSize, bSize do
        e_result[i]:copy(result[y])
        e_result[i + 1]:copy(result[y])
        e_errs[i]:copy(errs[y])
        e_errs[i + 1]:copy(errs[y])
        y = y + 1
    end
    local batched_inp = torch.ones(bSize * hSize, 3)
    local a_result = a:forward(batched_inp)
    local a_errs = a:backward(batched_inp, torch.range(1, 3):repeatTensor(6, 1))
    if not torch.all(torch.eq(a_errs, e_errs)) then
        print(a_errs)
        print(e_errs)
        print("Error: Backward pass does not match.")
    end
    if not torch.all(torch.eq(a_result, e_result)) then
        print(a_result)
        print(e_result)
        print("Error: Forward pass does not match.")
    end
end

function LstmTest:testCorrectForwardBackward()
    local w_const = 0.3
    local history = 4
    local a = nn.Lstm(1, 1, history)
    local b = nn.LSTM(1, 1)
    local a_params = a:getParameters()
    local b_params = b:getParameters()
    a_params:fill(w_const)
    b_params:fill(w_const)
    local i_a = torch.ones(history, 1)
    local i_b = torch.ones(1)
    local o_a = a:forward(i_a)
    local f_output = {}
    for i = 1, history do
        f_output[i] = b:forward(i_b)
    end
    local o_b = torch.cat(f_output):view(history, 1)
    local e_a = a:backward(i_a, torch.ones(history, 1))
    local b_output = {}
    for i = 1, history do
        b_output[i] = b:backward(i_b, torch.ones(1))
    end
    local e_b = torch.cat(b_output):view(history, 1)
    if torch.ne(o_a, o_b):sum() ~= 0 then
        print("Error: outputs are not consistent")
        return 1
    end

    for i = 1, history do
        assert(e_a[i]:eq(e_b[history + 1 - i]), "Backward errors do not match.")
    end

    return 0
end


BlstmTests = {}

function BlstmTests:testBatched()
    local iSize = 3
    local oSize = 6
    local hSize = 3
    local bSize = 2
    local b = nn.Blstm(iSize, oSize, hSize)
    local a = nn.Blstm(iSize, oSize, hSize)
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
    local a_errs = a:backward(batched_inp, torch.range(1, oSize):repeatTensor(6, 1))
    if not torch.all(torch.eq(a_errs, e_errs)) then
        print(a_errs)
        print(e_errs)
        print("Error: Backward pass does not match.")
    end
    if not torch.all(torch.eq(a_result, e_result)) then
        print(a_result)
        print(e_result)
        print("Error: Forward pass does not match.")
    end
end

GruTest = {}

function GruTest:testBatched()
    local iSize = 3
    local oSize = 3
    local hSize = 3
    local bSize = 2
    local b = nn.Gru(iSize, oSize, hSize)
    local a = nn.Gru(iSize, oSize, hSize)
    local x_b, _ = b:getParameters()
    local x_a, _ = a:getParameters()
    x_b:copy(x_a)
    local inp = torch.ones(hSize, iSize)
    local result = b:forward(inp):clone()
    local errs = b:backward(inp, torch.range(1, oSize):repeatTensor(hSize, 1))
    local e_errs = torch.Tensor(bSize * hSize, oSize)
    local e_result = e_errs:clone()
    local y = 1
    for i = 1, bSize * hSize, bSize do
        e_result[i]:copy(result[y])
        e_result[i + 1]:copy(result[y])
        e_errs[i]:copy(errs[y])
        e_errs[i + 1]:copy(errs[y])
        y = y + 1
    end
    local batched_inp = torch.ones(bSize * hSize, 3)
    local a_result = a:forward(batched_inp)
    local a_errs = a:backward(batched_inp, torch.range(1, 3):repeatTensor(6, 1))
    if not torch.all(torch.eq(a_errs, e_errs)) then
        print(a_errs)
        print(e_errs)
        print("Error: Backward pass does not match.")
    end
    if not torch.all(torch.eq(a_result, e_result)) then
        print(a_result)
        print(e_result)
        print("Error: Forward pass does not match.")
    end
end

function GruTest:testCorrectForwardBackward()
    local w_const = 0.3
    local history = 4
    local a = nn.Gru(1, 1, history)
    local b = nn.GRU(1, 1)
    local a_params = a:getParameters()
    local b_params = b:getParameters()
    a_params:fill(w_const)
    b_params:fill(w_const)
    local i_a = torch.ones(history, 1)
    local i_b = torch.ones(1)
    local o_a = a:forward(i_a)
    local f_output = {}
    for i = 1, history do
        f_output[i] = b:forward(i_b)
    end
    local o_b = torch.cat(f_output):view(history, 1)
    local e_a = a:backward(i_a, torch.ones(history, 1))
    local b_output = {}
    for i = 1, history do
        b_output[i] = b:backward(i_b, torch.ones(1))
    end
    local e_b = torch.cat(b_output):view(history, 1)
    tester:assertTensorEq(o_a, o_b, 0.0, "Outputs do not match")


    for i = 1, history do
        tester:assertTensorEq(e_a[i], e_b[history + 1 - i], 0.0,
            "Backward errors do not match.")
    end

    return 0
end


tester:add(LstmTest)
tester:add(BlstmTests)
tester:add(GruTest)


tester:run()
