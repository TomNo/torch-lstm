--[[
-- This module executes set of benchmarks against the standard rnn library
 ]]

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
require 'csvigo'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Benchmarking a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--output_file', "benchmarks.csv", "Where to store results.")
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
    table.insert(result, torch.type(modB))
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
        {nn.BiSequencer(nn.GRU(iSize, bOSize), nn.GRU(iSize, bOSize)), nn.Bgru(iSize, oSize, history)} }

    local addKey = function(key)
        if not results[key] then
            results[key] = {}
        end
    end

    for i=1, #bModules do
        local result = benchmark(bModules[i][1], bModules[i][2], iSize, history, nSeq, depth)
        local keyMem = result[1] .. "_" .. depth .. "_mem"
        local keyTime = result[1] .. "_" .. depth .. "_time"
        local keyMemOrig = keyMem .. "_eresearch"
        local keyTimeOrig = keyTime .. "_eresearch"
        addKey(keyMem)
        addKey(keyTime)
        addKey(keyMemOrig)
        addKey(keyTimeOrig)
        table.insert(results[keyMem], result[5])
        table.insert(results[keyTime], result[4])
        table.insert(results[keyMemOrig], result[3])
        table.insert(results[keyTimeOrig], result[2])

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


local iSize = 128
local oSize = 128
local nSeq = 16
local history = {100, 200, 300, 400, 500}
local depths = {1, 2, 3}

results = {["history"]=history}

for y=1, #depths do
    for i=1, #history do
        runBenchmark(iSize, oSize, history[i], nSeq, depths[y])
        print("\n")
    end
end

csvigo.save(params.output_file, results)
