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
cv_ds = Dataset(net.conf.train_file)
cv_data = train_ds:get()

net:train(tr_data, cv_data)

g_error, c_error = net:test(cv_data)
print("Error rate is: " .. g_error .. "%.")

print("training done")



