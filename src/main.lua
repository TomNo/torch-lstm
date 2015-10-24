require 'torch'
require 'NeuralNetwork'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--network_file','../network.jsn', 'neural network description file')
cmd:option('--config_file', '../config.cfg', 'training configuration file')
cmd:text()
params = cmd:parse(arg)

net = NeuralNetwork(params)
net:init()


