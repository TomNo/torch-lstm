require 'torch'
require 'NeuralNetwork'
require 'dataset-mnist'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--network_file','network.jsn', 'neural network description file')
cmd:option('--config_file', 'config.cfg', 'training configuration file')
cmd:option('--log_file', 'main.log', 'where to store log')
cmd:text()
params = cmd:parse(arg)
cmd:log(params.log_file, params)
net = NeuralNetwork(params, cmd)
net:init()
net:train(dataset)



