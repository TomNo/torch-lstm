require 'utils'
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
cmd:option('--forward_input', '', 'forward input file')
cmd:option('--config_file', '../timit_config.cfg', 'training configuration file')
cmd:option('--log_file', '', 'log file')
cmd:option('--output_model', 'final.model', 'name of the resulting serialized model')
cmd:option('--input_model', '', 'name of the resulting serialized model')
cmd:option('--log_softmax', true, 'apply LogSoftmax during forward pass')
cmd:text()
params = cmd:parse(arg)

if params.log_file == '' then
    params.log_file = "torch"
end
params.log_file = params.log_file .. utils.date() .. ".log"

cmd:log(params.log_file, params)
net = NeuralNetwork(params, cmd)
if params.input_model ~= '' then
    net.conf.model = params.input_model
end

if params.forward_input ~= '' then
    net.conf.test_file = params.forward_input
end

net:init()

if params.forward_pass then
    if params.forward_output == '' then
        error("Missing forward output file.")
    end
    if params.log_softmax then
        net:addLogSoftmax()
    end
    print("Computing forward pass for all sequences.")
    local testDs = TestSeqDs(net.conf.test_file)
    local outputDs = OutputDs(params.forward_output)
    while true do
        local seq = testDs:getSeq()
        if seq then
            outputDs:save(net:forward(seq.data, net.conf.truncate_seq / 4), seq.tag)
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
