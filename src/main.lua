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
cmd:option('--forward_pass', false, 'forward pass only')
cmd:option('--forward_output', '', 'resulting forward pass output file')
cmd:option('--config_file', '../timit_config.cfg', 'training configuration file')
cmd:option('--log_file', 'timit.log', 'log file')
cmd:option('--output_model', 'final.model', 'name of the resulting serialized model')
cmd:option('--input_model', '', 'name of the resulting serialized model')
cmd:text()
params = cmd:parse(arg)
cmd:log(params.log_file, params)
net = NeuralNetwork(params, cmd)
if params.input_model ~= '' then
    net.conf.model = params.input_model
end
net:init()

if params.forward_pass then
    if params.forward_output == '' then
        error("Missing forward output file.")
    end
    print("Computing forward pass for all sequences.")
    local testDs = TestSeqDs(net.conf.test_file)
    local outputDs = OutputDs(params.forward_output)
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

--eof
