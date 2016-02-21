require 'torch'
require 'nn'
require 'json'
require 'optim'
require 'EarlyStopping'
require 'Configuration'
require 'Lstm'
require 'Blstm'


-- TODO procesing sequences by uterrances
-- TODO baidu ctc https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
-- TODO forward pass is terribly slow

torch.setdefaulttensortype('torch.FloatTensor')


CLIP_MIN = -1.0
CLIP_MAX = 1.0

local function gradClip(element)
    if element > CLIP_MAX then
        return CLIP_MAX
    elseif element < CLIP_MIN then
        return CLIP_MIN
    else
        return element
    end
end


local NeuralNetwork = torch.class('NeuralNetwork')


NeuralNetwork.FEED_FORWARD_TANH = "feedforward_tanh"
NeuralNetwork.FEED_FORWARD_LOGISTIC = "feedforward_logistic"
NeuralNetwork.MULTICLASS_CLASSIFICATION = "multiclass_classification"
NeuralNetwork.FEED_FORWARD_RELU = "feedforward_relu"
NeuralNetwork.SOFTMAX = "softmax"
NeuralNetwork.INPUT = "input"
NeuralNetwork.LSTM = "lstm"
NeuralNetwork.BLSTM = "blstm"
NeuralNetwork.DEFAULT_HISTORY = 5


function NeuralNetwork:__init(params, log)
    for key, value in pairs(params) do
        if self.key then
            perror("Parameters already exists: " .. key)
        end
        self[key] = value
    end
    self.desc = nil -- should contain json net topology description
    self.output_size = nil -- size of the last layer
    self.input_size = nil -- size of the input layer
    self.conf = {} -- configuration regarding training
    self.m_params = nil -- model params
    self.m_grad_params = nil -- model gradients
    self.log = log
    self.log:addTime('NeuralNetwork', '%F %T')
end


function NeuralNetwork:init()
    print("Initializing neural network.")
    self.conf = Configuration.new(self.config_file)
    local f = assert(io.open(self.conf.network, "r"),
        "Coult not open the network file: " .. self.conf.network)
    local net_desc = f:read("*all") -- this is strange might be better way how to read whole file
    f:close()
    -- description just as in the currennt
    self.desc = json.decode(net_desc)
    self.output_size = self.desc.layers[#self.desc.layers - 1].size
    self.input_size = self.desc.layers[1].size
    if self.conf.cuda then
        local loadCuda = function()
            require 'cutorch'
            require 'cunn'
            local deviceId = os.getenv("CUDA_VISIBLE_DEVICES") or 1
            -- it is pretty confusing as sometime is enough to set the
            -- id from the variable and sometime it is necessary to iterate
            if deviceId == 0 then
                deviceId = deviceId + 1
            end
            print("Trying to set cuda device: " .. deviceId)
            cutorch.setDevice(deviceId)
        end
        local cudaEnabled = pcall(loadCuda)
        if not cudaEnabled then print("Could not load cuda.Proceeding anyway.") end
    end
    self.e_stopping = EarlyStopping.new(self.conf.max_epochs_no_best)
    if self.conf.model then
        self:loadModel(self.conf.model)
        self:_addCriterion(self.desc.layers[#self.desc.layers])
    else
        self.model = nn.Sequential()
        self:_createLayers()
        if self.conf.weights then
            self:loadWeights(self.conf.weights)
        else
            self.model:reset(self.conf.weights_uniform_max)
        end
    end

    self.m_params, self.m_grad_params = self.model:getParameters()
    print("Model contains " .. self.m_params:size(1) .. " weights.")
    print("Model:")
    print(self.model)
    if self.conf.optimizer == "rmsprop" then
        self.optim = optim.rmsprop
    elseif self.conf.optimizer == "adadelta" then
        self.optim = optim.adadelta
    else
        self.optim = optim.sgd
    end
end

function NeuralNetwork:_addCriterion(layer)
    if layer.type == NeuralNetwork.MULTICLASS_CLASSIFICATION then
        self.criterion = nn.ClassNLLCriterion()
        if self.conf.cuda then
            self.criterion = self.criterion:cuda()
        end
    else
        error("Unknown objective function " .. layer["type"] .. ".")
    end
end


function NeuralNetwork:_createLayers()
    if self.desc.layers == nil then
        error("Missing layers section in " .. self.network_file .. ".")
    end

    for index, layer in ipairs(self.desc.layers) do
        if index > 1 and index ~= #self.desc.layers then
            if layer.name == nil or layer.bias == nil or layer.size == nil or layer.type == nil then
                error("Layer number: " .. index .. " is missing required attribute.")
            end
            self:_addLayer(self.desc.layers[index], self.desc.layers[index - 1])
        elseif index == #self.desc.layers then -- last layer is objective function
            if layer.type == nil then
                error("Layer number: " .. index .. " is missing required attribute.")
            end
            self:_addCriterion(layer)
        end
    end
    if self.conf.cuda then
        self.model = self.model:cuda()
    end
end


function NeuralNetwork:_addLayer(layer, p_layer)
    if p_layer == nil then error("Missing previous layer argument.") end
    if layer.type == NeuralNetwork.INPUT then
        return
    elseif layer.type == NeuralNetwork.FEED_FORWARD_LOGISTIC then
        self.model:add(nn.Linear(p_layer.size, layer.size))
        self.model:add(nn.Sigmoid())
    elseif layer.type == NeuralNetwork.FEED_FORWARD_TANH then
        self.model:add(nn.Linear(p_layer.size, layer.size))
        self.model:add(nn.Tanh())
    elseif layer.type == NeuralNetwork.FEED_FORWARD_RELU then
        self.model:add(nn.Linear(p_layer.size, layer.size))
        self.model:add(nn.ReLU())
    elseif layer.type == NeuralNetwork.LSTM then
        self.model:add(nn.Lstm(p_layer.size, layer.size, self.conf.truncate_seq, layer.batch_normalization))
    elseif layer.type == NeuralNetwork.BLSTM then
        self.model:add(nn.Blstm(p_layer.size, layer.size, self.conf.truncate_seq, layer.batch_normalization))
    elseif layer.type == NeuralNetwork.SOFTMAX then
        self.model:add(nn.Linear(p_layer.size, layer.size))
        self.model:add(nn.LogSoftMax())
    else
        error("Unknown layer type: " .. layer.type ".")
    end
    if layer.dropout and layer.dropout > 0 then
        self.model:add(nn.Dropout(layer.dropout))
    end
--    if layer.batch_normalization then
--        self.model:add(nn.BatchNormalization(layer.size))
--    end
end


function NeuralNetwork:_calculateError(predictions, labels)
    local _, preds = predictions:max(2)
    return preds:typeAs(labels):ne(labels):sum()
end


function NeuralNetwork:train(dataset, cv_dataset)
    if self.conf.cuda then
        print("Training on GPU.")
    else
        print("Training on CPU.")
    end
    assert(dataset.cols == self.input_size,
        "Dataset input does not fit first layer size.")
    local opt_params = {
        learningRate = self.conf.learning_rate,
        weightDecay = self.conf.weight_decay,
        momentum = self.conf.momentum,
        learningRateDecay = self.conf.learning_rate_decay
    }
    local state = {}
    for epoch = 1, self.conf.max_epochs do
        self.model:training()
        print('==> doing epoch ' .. epoch .. ' on training data.')
        local time = sys.clock()
        dataset:startBatchIteration(self.conf.parallel_sequences,
            self.conf.truncate_seq,
            self.conf.shuffle_sequences,
            self.conf.random_shift)
        local e_error = 0
        local e_predictions = 0
        local i_count = 0
        local grad_clips_accs = 0
        while true do
            self.inputs, self.labels = dataset:nextBatch()
            if self.inputs == nil then
                break
            end
            local feval = function(x)
                collectgarbage()
                -- get new parameters
                if self.m_params ~= x then
                    self.m_params:copy(x)
                end
                -- reset gradients
                self.m_grad_params:zero()
                local outputs = self.model:forward(self.inputs)
                local err = self.criterion:forward(outputs, self.labels)
                --        err = err/self.inputs:size(1)
                e_error = e_error + err
                e_predictions = e_predictions + self:_calculateError(outputs, self.labels)
                i_count = i_count + outputs:size(1)
                self.model:backward(self.inputs, self.criterion:backward(outputs, self.labels))
                -- apply gradient clipping
                self.m_grad_params:clamp(CLIP_MIN, CLIP_MAX)
                grad_clips_accs = self.m_grad_params:eq(1):cat(self.m_grad_params:eq(-1)):sum() + grad_clips_accs
                --        print("Max gradient: " .. self.m_grad_params:max())
                --        print("Min gradient: " .. self.m_grad_params:min())
                --        print("Average gradient: " .. self.m_grad_params:sum() / self.m_grad_params:nElement())
                return err, self.m_grad_params
            end -- feval
            self.optim(feval, self.m_params, opt_params, state)
        end -- mini batch
        collectgarbage()
        print("Epoch has taken " .. sys.clock() - time .. " seconds.")
        print("Gradient clippings occured " .. grad_clips_accs)
        grad_clips_accs = 0
        print(string.format("Error rate on training set is: %.2f%% and loss is: %.4f",
            e_predictions / i_count * 100, e_error))
        --autosave model or weights
        if self.conf.autosave_model then
            local prefix = ""
            if self.conf.autosave_prefix then
                prefix = self.conf.autosave_prefix .. "_"
            end
            self:saveModel(prefix .. "epoch_" .. epoch .. ".model")
        end

        if self.conf.autosave_weights then
            local prefix = ""
            if self.conf.autosave_prefix then
                prefix = self.conf.autosave_prefix .. "_"
            end
            self:saveWeights(prefix .. "epoch_" .. epoch .. ".weights")
        end

        if not self.conf.validate_every or epoch % self.conf.validate_every == 0 then
            if cv_dataset and not self.e_stopping:validate(self, cv_dataset) then
                print("No lowest validation error was reached -> stopping training.")
                self.m_params:copy(self.e_stopping:getBestWeights())
                break
            end
        end
    end -- epoch
    print("Training finished.")
end


-- put whole sequence in one batch
function NeuralNetwork:forward(data)
    assert(data:size(2) == self.input_size,
        "Dataset input does not match first layer size.")
    self.model:evaluate()
    local bCount = math.floor(data:size(1)/self.conf.truncate_seq)
    local overhang = data:size(1) % self.conf.truncate_seq
    local iCount = self.conf.truncate_seq * bCount
    local outputs = torch.Tensor(data:size(1), self.output_size)
    local tmp = data[{{1, iCount}}]:clone()
    local inputs = tmp:reshape(bCount, self.conf.truncate_seq, data:size(2)):transpose(1,2):reshape(iCount, data:size(2))
    if self.conf.cuda then
        inputs = inputs:cuda()
    end
    local tOutput = self.model:forward(inputs):reshape(self.conf.truncate_seq, bCount, self.output_size)
    outputs[{{1, iCount}}]:copy(tOutput:transpose(1,2):reshape(iCount, self.output_size))
    if overhang > 0 then
        local bIndex = data:size(1) - self.conf.truncate_seq + 1
        inputs:resize(self.conf.truncate_seq, data:size(2))
        inputs:copy(data[{{bIndex, data:size(1)}}])
        local tOutput = self.model:forward(inputs)
        outputs[{{bIndex, data:size(1)}}]:copy(tOutput)
    end
    collectgarbage()
    return outputs
end


function NeuralNetwork:test(dataset)
    assert(dataset.cols == self.input_size, "Dataset inputs does not match first layer size.")
    local g_error = 0
    local c_error = 0
    local i_count = 0
    self.model:evaluate()
    dataset:startBatchIteration(self.conf.parallel_sequences,
                                self.conf.truncate_seq)
    while true do
        self.inputs, self.labels = dataset:nextBatch()
        if self.inputs == nil then
            break
        end
        local output = self.model:forward(self.inputs)
        i_count = i_count + output:size(1)
        c_error = c_error + self.criterion(output, self.labels)
        g_error = g_error + self:_calculateError(output, self.labels)
    end
    collectgarbage()
    return (g_error / i_count) * 100, c_error
end


function NeuralNetwork:saveWeights(filename)
    print("Saving weights into: " .. filename)
    torch.save(filename, self.m_params)
end


function NeuralNetwork:loadWeights(filename)
    print("Loading weights from file: " .. filename)
    self.m_params:copy(torch.load(filename))
end


function NeuralNetwork:saveModel(filename)
    print("Saving model to " .. filename)
    self.model:clearState()
    torch.save(filename, self.model)
end


function NeuralNetwork:loadModel(filename)
    print("Loading model from " .. filename)
    self.model = torch.load(filename)
    self.m_params, self.m_grad_params = self.model:getParameters()
end


--eof
