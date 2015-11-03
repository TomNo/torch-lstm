require 'torch'
require 'NeuralNetwork'
require 'Dataset'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--network_file','../timit_network.jsn', 'neural network description file')
cmd:option('--config_file', '../timit_config.cfg', 'training configuration file')
cmd:option('--log_file', 'timit.log', 'log file')
cmd:text()
params = cmd:parse(arg)
cmd:log(params.log_file, params)
net = NeuralNetwork(params, cmd)
net:init()
train_ds = Dataset(net.conf.train_file)
tr_data = train_ds:get()
cv_ds = Dataset(net.conf.val_file)
cv_data = cv_ds:get()
test_ds = Dataset(net.conf.test_file)
test_data = test_ds:get()
net:train(tr_data, cv_data)
net:saveModel('timit_network_trained')
local output = net:forward(test_data)
output_ds = Dataset('timit_result')
output_ds:save(output, test_data.tags)
print("training done")



