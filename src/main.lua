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

--local Bum = torch.class('Bum')
--DBum , parent= torch.class('DBum', 'Bum')
--
--Bum.A = 50
--
--function Bum:__init(a,b)
--  self.a = a
--  self.b = b
--end
--
--function Bum:printMe()
--  print(self.a)
--  print(self.b)
--end
--
--function DBum:__init(a,b)
--  parent.__init(self,a,b)
--  self.a = 8
--  self.A = 5
--  self:printMe()
--end
--
--a = DBum.new(5,6)
--print(DBum.A)
