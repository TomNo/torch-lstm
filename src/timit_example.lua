require 'torch'
require 'NeuralNetwork'
require 'TrainSeqDs'
require 'TestSeqDs'
require 'OutputDs'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('--network_file', '../timit_network.jsn', 'neural network description file')
cmd:option('--forward_pass', false, 'forward pass only')
cmd:option('--forward_output', '', 'resulting forward pass output file')
cmd:option('--trained_network', 'timit_network_trained', 'load trained model')
cmd:option('--config_file', '../timit_config.cfg', 'training configuration file')
cmd:option('--log_file', 'timit.log', 'log file')
cmd:option('--output_model', 'timit_network_trained', 'name of the resulting trained model')
cmd:text()
params = cmd:parse(arg)
cmd:log(params.log_file, params)
net = NeuralNetwork(params, cmd)
net:init()

if params.forward_pass then
    if params.forward_output == '' then
        error("Missing forward output file.")
    end

    local testDs = TestSeqDs(net.conf.test_file)
    local outputDs = OutputDs(params.forward_output)
    net:loadModel(params.trained_network)
    while true do
        local seq = testDs:getSeq()
        if seq then
            outputDs:save(net:forward(seq.data), seq.tag)
        else
            break
        end
    end
    outputDs:close()
    print("Forward pass was saved to the: " .. params.forward_output)
else
    local train_ds = TrainSeqDs(net.conf.train_file, net.conf.cuda, true)
    local val_ds = TrainSeqDs(net.conf.val_file, net.conf.cuda, true)
    net:train(train_ds, val_ds)
    net:saveModel(params.output_model)
end


--train_ds = TrainSeqDs(net.conf.train_file, net.conf.cuda)

--cv_ds = TrainDs(net.conf.val_file)
--cv_data = cv_ds:get(4096)
--test_ds = TestDs(net.conf.test_file)
--test_data = test_ds:get(4096)


--local output = net:forward(test_data)
--output_ds = TestDs('timit_result')
--output_ds:save(output, test_data.tags)
--print("training done")



