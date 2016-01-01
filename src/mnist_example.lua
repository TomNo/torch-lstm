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
nbTrainingPatches = 60000
nbTestingPatches = 10000
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)
--
--
train_features = torch.Tensor(nbTrainingPatches, 32*32)
test_features = torch.Tensor(nbTestingPatches, 32*32)
train_labels = torch.Tensor(nbTrainingPatches)
test_labels = torch.Tensor(nbTestingPatches)

local c_error = 0
local g_error = 0
for i=1,testData:size() do
  local label = testData[i][2]
  local _, m = label:max(1)
  test_features[i] = testData[i][1]:view(testData[i][1]:nElement())
  test_labels[i] = m
end

for i=1,trainData:size() do
  local label = trainData[i][2]
  local _, m = label:max(1)
  train_features[i] = trainData[i][1]:view(trainData[i][1]:nElement())
  train_labels[i] = m
end

test_dataset = {["features"]=test_features, ["labels"]=test_labels}
test_dataset.size = function() return nbTrainingPatches end
test_dataset.cols = 1024
test_dataset.rows = 10000

train_dataset = {["features"]=train_features, ["labels"]=train_labels}
train_dataset.size = function() return nbTestingPatches end
train_dataset.cols = 1024
train_dataset.rows = 60000

g_error, c_error = net:test(test_dataset)
print("Error rate is: " .. g_error .. "%.")

net:train(train_dataset)

g_error, c_error = net:test(test_dataset)
print(net.model)
print("Error rate is: " .. g_error .. "%.")

print("training done")



