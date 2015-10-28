require 'torch'
require 'NeuralNetwork'
require 'dataset-mnist'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--network_file','../mnist_network.jsn', 'neural network description file')
cmd:option('--config_file', '../mnist_config.cfg', 'training configuration file')
cmd:option('--log_file', 'mnist.log', 'log file')
cmd:text()
params = cmd:parse(arg)
cmd:log(params.log_file, params)
net = NeuralNetwork(params, cmd)
net:init()

-- mnist testing
geometry = {32,32}
nbTrainingPatches = 5000
nbTestingPatches = 1000
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)
--
--
test_dataset = {}
local c_error = 0
local g_error = 0
for i=1,testData:size() do
  local label = testData[i][2]
  local _, m = label:max(1)
  test_dataset[i] = {[1]=testData[i][1]:view(testData[i][1]:nElement()), [2]=m}
end

function test_dataset:size()
  return #test_dataset
end
g_error, c_error = net:test(test_dataset)
print("Error rate is: " .. g_error .. "%.")

dataset = {}
for i=1,trainData:size() do
  local label = trainData[i][2]
  local _, m = label:max(1)
  dataset[i] = {[1]=trainData[i][1]:view(trainData[i][1]:nElement()), [2]=m}
end

function dataset:size()
  return #dataset
end

net:train(dataset)

g_error, c_error = net:test(test_dataset)
print(net.model)
print("Error rate is: " .. g_error .. "%.")

print("training done")



