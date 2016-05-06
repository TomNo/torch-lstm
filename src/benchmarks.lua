require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'rnn'
require 'Lstm'
require 'Gru'
require 'Blstm'
require 'Bgru'
require 'utils'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Benchmarking a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--input_size', 512, 'Input layer size')
cmd:option('--output_size', 512, 'Output layer size')
cmd:option('--depth', 1, 'Depth of benchmarked network')
cmd:option('--history', 100, 'Length of the sequences')
cmd:option('--nseq', 16, 'Sequence count')
cmd:text()
params = cmd:parse(arg)

-- comparsion of implemented modules with ElementResearch implementation

--[[
-- Benchmark two modules, print training time and memory consumption
-- modA    - ElementResearch module
-- modB    - own implementation of the same module
-- iSize   - input size
-- history - number of timestep in sequence
-- nSeq    - number of concurently executed sequences
 ]]
local function benchmark(modA, modB, iSize, history, nSeq, depth)
    local input = torch.range(1, iSize * history * nSeq):reshape(history*nSeq, iSize):cuda()
    local result = {}
    local sInput = {}
    local sIndex = 1
    for _=1, history do
        table.insert(sInput, input[{{sIndex, sIndex + nSeq - 1}}])
        sIndex = sIndex + nSeq
    end
    local cMem = cutorch.getMemoryUsage()
    local a = nn.Sequential()
    for _=1, depth do
        a:add(modA:clone())
    end
    a = a:cuda()
    table.insert(result, torch.type(modB))
    local sTime = os.clock()
    a:forward(sInput)
    a:backward(sInput, sInput)
    table.insert(result, os.clock() - sTime)
    local fCMem = cutorch.getMemoryUsage()
    local memoryConcumption = (cMem - fCMem) / math.pow(2,20)
    a = nil
    collectgarbage()
    table.insert(result, memoryConcumption)
    local cMem = cutorch.getMemoryUsage()
    local b = nn.Sequential()
    for _=1, depth do
        b:add(modB:clone())
    end
    b = b:cuda()
    local bSizes = {}
    for _=1, history do
        table.insert(bSizes, nSeq)
    end
    b:apply(function (m) m.bSizes = bSizes end)
    local sTime = os.clock()
    b:forward(input)
    b:backward(input, input)
    table.insert(result, os.clock() - sTime)
    local fCMem = cutorch.getMemoryUsage()
    local memoryConcumption = (cMem - fCMem) / math.pow(2,20)
    table.insert(result, memoryConcumption)
    b = nil
    collectgarbage()
    return result
end


function runBenchmark(iSize, oSize, history, nSeq, depth)
    print(string.format("Benchmark results for following parameters:"))
    print(string.format("input size: %s ,output size: %s, history: %s, depth=%s, %s sequences",
         iSize, oSize, history, depth, nSeq))
    print("               | ElementResearch   | Own implementation")
    print("Module type    | Time    |  Memory | Time    |  Memory ")

    local bOSize = oSize / 2

    local bModules = {{nn.Sequencer(nn.LSTM(iSize, oSize)), nn.Lstm(iSize, oSize, history)},
        {nn.Sequencer(nn.GRU(iSize, oSize)), nn.Gru(iSize, oSize, history)},
        {nn.BiSequencer(nn.LSTM(iSize, bOSize), nn.LSTM(iSize, bOSize)), nn.Blstm(iSize, oSize, history)},
        {nn.BiSequencer(nn.GRU(iSize, bOSize), nn.GRU(iSize, bOSize)), nn.Bgru(iSize, oSize, history)}}


    for i=1, #bModules do
        local result = benchmark(bModules[i][1], bModules[i][2], iSize, history, nSeq, depth)
        for y=1, #result do
            result[y] = result[y] .. "                         "
        end

        print(string.format("%s| %s s| %s MB| %s s| %s MB",
            string.sub(result[1], 1 , 15), string.sub(result[2], 1 , 6),
            string.sub(result[3], 1 , 5), string.sub(result[4], 1 , 6),
            string.sub(result[5], 1 , 5)))
        collectgarbage()
    end
end


--local iSize = 128
--local oSize = 128
--local nSeq = 16
--local history = {100, 200, 300, 400, 500}
--local depths = {1, 2, 3}

--for y=1, #depths do
--    for i=1, #history do
--        runBenchmark(iSize, oSize, history[i], nSeq, depths[y])
--        print("\n")
--    end
--end
runBenchmark(params.input_size, params.output_size, params.history, params.nseq, params.depth)










