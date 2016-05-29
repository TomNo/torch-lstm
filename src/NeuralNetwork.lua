require 'utils'
require 'torch'
require 'nn'
require 'json'
require 'optim'
require 'EarlyStopping'
require 'Configuration'
require 'Lstm'
require 'Blstm'
require 'Gru'
require 'Bgru'
require 'RecLayer'
require 'CtcCriterion'
require 'rmsprop'
require 'ParallelTable'
require 'FFLayer'


-- TODO rewrite using nngraph

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


local actTypes = {
    tanh = nn.Tanh,
    logistic = nn.Sigmoid,
    relu = nn.ReLU,
    prelu = nn.PReLU
}

local function getActType(lType)
    local aType
    local result = {}
    for i in lType:gsub("_", " "):gmatch("%S+") do
        table.insert(result, i)
    end
    return actTypes[result[2]]
end


local NeuralNetwork = torch.class('NeuralNetwork')

NeuralNetwork.FEED_FORWARD = "feedforward"
NeuralNetwork.REC = "rec"
NeuralNetwork.IREC = "irec"
NeuralNetwork.B_REC = "brec"
NeuralNetwork.B_IREC = "birec"
NeuralNetwork.MULTICLASS_CLASSIFICATION = "multiclass_classification"
NeuralNetwork.CTC = "ctc"
NeuralNetwork.LINEAR = "linear"
NeuralNetwork.INPUT = "input"
NeuralNetwork.LSTM = "lstm"
NeuralNetwork.BLSTM = "blstm"
NeuralNetwork.GRU = "gru"
NeuralNetwork.BGRU = "bgru"
NeuralNetwork.DEFAULT_HISTORY = 5
NeuralNetwork.CTC_CRITERION = "nn.CtcCriterion"


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
    if log then
        self.log = log
        self.log:addTime('NeuralNetwork', '%F %T')
    end
    self.conf = Configuration.new(self.config_file)
end


function NeuralNetwork:init()
    print("Initializing neural network.")
    local f = assert(io.open(self.conf.network, "r"),
        "Coult not open the network file: " .. self.conf.network)
    local net_desc = f:read("*all") -- this is strange might be better way how to read whole file
    f:close()
    -- description just as in the currennt
    self.desc = json.decode(net_desc)
    self.output_size = self.desc.layers[#self.desc.layers - 1].size
    self.input_size = self.desc.layers[1].size
    self.inputs = torch.Tensor()
    self.labels = torch.Tensor()
    if self.conf.cuda then
        local loadCuda = function()
            --cudnn tanh cannot handle non-contingenouse arrays
            require 'cutorch'
            require 'cunn'
            -- cudnn consumes more memory **TODO** investigate
            --            require 'cudnn'
            local _, aMem = cutorch.getMemoryUsage()
            print("Available memory is: " .. aMem)
        end
        local cudaEnabled = pcall(loadCuda)
        if not cudaEnabled then print("Could not load cuda.Proceeding anyway.") end
        self.inputs = self.inputs:cuda()
        self.labels = self.labels:cuda()
    end
    self.e_stopping = EarlyStopping.new(self.conf.max_epochs_no_best)
    if self.conf.model then
        self:loadModel(self.conf.model)
        self:_addCriterion(self.desc.layers[#self.desc.layers])
    else
        self.model = nn.Sequential()
        self.model.forward = function(mod, input, sizes)
            local bSizes = utils.getBatchSizes(sizes)
            mod:apply(function(m) m.bSizes = bSizes end)
            return mod:updateOutput(input)
        end
        self:_createLayers()
        if cudnn then
            cudnn.convert(self.model, cudnn)
        end
        -- inspired by fb resnet
        local cache = {}
        self.model:apply(function(m)
            local moduleType = torch.type(m)
            if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
                if cache[moduleType] == nil then
                    if self.conf.cuda then
                        cache[moduleType] = torch.CudaStorage(1)
                        m.gradInput = torch.CudaTensor(cache[moduleType], 1, 0)
                    else
                        cache[moduleType] = torch.FloatStorage(1)
                        m.gradInput = torch.FloatTensor(cache[moduleType], 1, 0)
                    end
                end

            end
        end)
        for i, m in ipairs(self.model:findModules('nn.ConcatTable')) do
            if self.conf.cuda then
                if cache[i % 2] == nil then
                    cache[i % 2] = torch.CudaStorage(1)
                end
                m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
            else
                if cache[i % 2] == nil then
                    cache[i % 2] = torch.FloatStorage(1)
                end
                m.gradInput = torch.FloatTensor(cache[i % 2], 1, 0)
            end
        end
    end
    self.m_params, self.m_grad_params = self.model:getParameters()

    if self.conf.weights then
        self:loadWeights(self.conf.weights)
    end

    if not self.conf.model and not self.conf.weights then
        self.model:reset(self.conf.weights_uniform_max)
    end
    print("Model contains " .. self.m_params:size(1) .. " weights.")
    print("Model:")
    print(self.model)
    print("Criterion:")
    print(self.criterion)
    if self.conf.optimizer == "rmsprop" then
        self.optim = optim.rmsprop
    elseif self.conf.optimizer == "g_rmsprop" then
        self.optim = optim.grmsprop
    elseif self.conf.optimizer == "adadelta" then
        self.optim = optim.adadelta
    else
        self.optim = optim.sgd
    end
end

function NeuralNetwork:_addCriterion(layer)
    if layer.type == NeuralNetwork.MULTICLASS_CLASSIFICATION then
        self.criterion = nn.CrossEntropyCriterion()
    elseif layer.type == NeuralNetwork.CTC then
        self.criterion = nn.CtcCriterion(self.conf.history)
    else
        error("Unknown objective function " .. layer["type"] .. ".")
    end
    if self.conf.cuda then
        self.criterion = self.criterion:cuda()
    end
end


function NeuralNetwork:_createLayers()
    if self.desc.layers == nil then
        error("Missing layers section in " .. self.network_file .. ".")
    end

    for index, layer in ipairs(self.desc.layers) do
        if index > 1 and index ~= #self.desc.layers then
            if layer.size == nil or layer.type == nil then
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
    local aType = getActType(layer.type)
    local nLayer = nil
    if layer.type == NeuralNetwork.INPUT then
        return
        -- feedforward
    elseif string.match(layer.type, NeuralNetwork.FEED_FORWARD) then
        nLayer = nn.FFLayer(p_layer.size, layer.size, aType,
            layer.batch_normalization, layer.dropout)
        -- bidirectional identity rnn
    elseif string.match(layer.type, NeuralNetwork.B_IREC) then
        nLayer = nn.BIRecLayer(aType, p_layer.size, layer.size, self.conf.history,
              layer.batch_normalization, layer.dropout)
        -- identity rnn
    elseif string.match(layer.type, NeuralNetwork.IREC) then
        nLayer = nn.IRecLayer(aType, p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- bidirectional rnn
    elseif string.match(layer.type, NeuralNetwork.B_REC) then
        nLayer = nn.BiRecLayer(aType, p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- classic rnn
    elseif string.match(layer.type, NeuralNetwork.REC) then
        nLayer = nn.RecLayer(aType, p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- lstm
    elseif layer.type == NeuralNetwork.LSTM then
        nLayer = nn.Lstm(p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- blstm
    elseif layer.type == NeuralNetwork.BLSTM then
        nLayer = nn.Blstm(p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- gru
    elseif layer.type == NeuralNetwork.GRU then
        nLayer = nn.Gru(p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- bgru
    elseif layer.type == NeuralNetwork.BGRU then
        nLayer = nn.Bgru(p_layer.size, layer.size, self.conf.history,
            layer.batch_normalization, layer.dropout)
        -- linear layer
    elseif layer.type == NeuralNetwork.LINEAR then
        nLayer = nn.Linear(p_layer.size, layer.size)
    else
        error("Unknown layer type: " .. layer.type ".")
    end
    self.model:add(nLayer)
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
        nesterov = self.conf.nesterov_momentum
    }
    local state = {}
    if self.conf.optimizer_state then
        print("Loading optimizer state from file: " .. self.conf.optimizer_state)
        state = torch.load(self.conf.optimizer_state)
    end

    -- if whole dataset in the memory, make weights of the criterion
    -- inversely proportional to the label frequency
    if dataset.a_labels and self.criterion.total_weight_tensor then
        local critWeights = dataset.a_labels:histc(self.output_size)
        critWeights:mul(self.output_size)
        critWeights:div(dataset.a_labels:size(1))
        self.criterion.total_weight_tensor:resize(self.output_size)
        self.criterion.total_weight_tensor:copy(critWeights)
    end

    for epoch = 1, self.conf.max_epochs do
        self.model:training()
        collectgarbage()
        print('==> doing epoch ' .. epoch .. ' on training data.')
        local time = sys.clock()
        local cLabels = true
        if torch.type(self.criterion) == self.CTC_CRITERION then
            cLabels = false
        end
        dataset:startBatchIteration(self.conf.sampling_type,
            self.conf.batch_size,
            self.conf.shuffle,
            self.conf.history,
            self.conf.split_sequences, cLabels)
        local e_error = 0
        local e_predictions = 0
        local i_count = 0
        local grad_clips_accs = 0
        local b_count = 0
        local g_error = 0
        while true do
            local inputs, labels, sizes = dataset:nextBatch()
            if inputs == nil then
                break
            end
            self:_setTrainData(inputs, labels)
            b_count = b_count + 1
            local feval = function(x)
                collectgarbage()
                -- get new parameters
                if self.m_params ~= x then
                    self.m_params:copy(x)
                end
                -- reset gradients
                self.m_grad_params:zero()
                local outputs = self.model:forward(self.inputs, sizes)
                g_error = g_error + self:_calculateError(outputs, self.labels)
                local err = self.criterion:forward(outputs, self.labels, sizes,
                                                   self.model.bSizes)
                e_error = e_error + err
                i_count = i_count + utils.sumTable(sizes)
                local cGradInput = self.criterion:backward(outputs, self.labels)
                self.model:backward(self.inputs, cGradInput)
                -- apply gradient clipping
                self.m_grad_params:clamp(CLIP_MIN, CLIP_MAX)
                if self.conf.verbose then
                    grad_clips_accs = self.m_grad_params:eq(1):cat(self.m_grad_params:eq(-1)):sum() + grad_clips_accs
                    print("Max gradient: " .. self.m_grad_params:max())
                    print("Min gradient: " .. self.m_grad_params:min())
                    print("Average gradient: " .. self.m_grad_params:sum() / self.m_grad_params:nElement())
                end
                return err, self.m_grad_params
            end -- feval
            self.optim(feval, self.m_params, opt_params, state)
        end -- mini batch
        collectgarbage()
        print("Epoch has taken " .. sys.clock() - time .. " seconds.")
        if self.conf.verbose then
            print("Gradient clippings occured " .. grad_clips_accs)
            grad_clips_accs = 0
        end
        e_error = e_error / b_count
        print(string.format("Loss on training set is: %.4f", e_error))
        print(string.format("Error on training set set is: %.2f%%",
                            g_error / i_count * 100))
        if self.conf.learning_rate_decay and epoch % self.conf.decay_every == 0 then
            local nLr = opt_params.learningRate * self.conf.learning_rate_decay
            local mLr = self.conf.min_learning_rate
            if (mLr and nLr > mLr) or not mLr then
                opt_params.learningRate = nLr
                print("Learning rate after decay is: " .. opt_params.learningRate)
            else
                print("Not decaying learning rate as it is less than minimal learning rate.")
                opt_params.learningRate = mLr
                print("Setting learning rate to minimal learning rate: " .. opt_params.learningRate)
            end
        end

        --autosave model, weights, optimizer
        if self.conf.autosave_optimizer then
            local prefix = ""
            if self.conf.autosave_prefix then
                prefix = self.conf.autosave_prefix .. "_"
            end
            local oString = string.format("%sepoch_%s%s.optimizer", prefix,
                epoch, utils.date())
            print("Saving optimizer to: " .. oString)
            torch.save(oString, state)
        end

        if self.conf.autosave_model then
            local prefix = ""
            if self.conf.autosave_prefix then
                prefix = self.conf.autosave_prefix .. "_"
            end
            self:saveModel(prefix .. "epoch_" .. epoch .. utils.date() .. ".model")
        end

        if self.conf.autosave_weights then
            local prefix = ""
            if self.conf.autosave_prefix then
                prefix = self.conf.autosave_prefix .. "_"
            end
            self:saveWeights(prefix .. "epoch_" .. epoch .. utils.date() .. ".weights")
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


function NeuralNetwork:_calculateError(predictions, labels)
    local result
    if torch.type(self.criterion) == self.CTC_CRITERION then
        local aOutput = {}
        local aLabels = predictions:max(2)
        local pSymbol = -1
        for i=1, aLabels:size(1) do
            if aLabels[i] ~= 1 and aLabels[i] ~= pSymbol then
                table.insert(aOutput, aLabels[i])
                pSymbol = aLabels[i]
            end
            utils.levens
        end

        utils.leven
    else
        local _, preds = predictions:max(2)
        result = preds:typeAs(labels):ne(labels):sum()
    end
    return result
end

-- put whole sequence in one batch
-- TODO put parallel seq in one batch
-- TODO cpu forward pass is not supported
function NeuralNetwork:forward(data, overlap)
    assert(data:size(2) == self.input_size,
        "Dataset input does not match first layer size.")
    self.model:evaluate()

    overlap = overlap or 0
    if data:size(1) <= self.conf.history then
        return self.model:forward(data:cuda(), { data:size(1) }):float()
    end
    local step = self.conf.history - 2 * overlap

    local iSeqs = math.floor((data:size(1) - 2 * overlap) / step)
    local overhang = data:size(1) % step

    local input = torch.Tensor(iSeqs * self.conf.history, data:size(2))
    local sizes = {}
    for i = 1, iSeqs do
        table.insert(sizes, self.conf.history)
    end
    for y = 1, self.conf.history do
        for i = 1, iSeqs do
            input[(y - 1) * iSeqs + i] = data[(i - 1) * step + y]
        end
    end

    input = input:cuda()

    local output = torch.Tensor(data:size(1), self.output_size)
    --calculate first the end of the sequence
    local eInt = { { data:size(1) - self.conf.history + 1, data:size(1) } }
    output[eInt]:copy(self.model:forward(data[eInt]:cuda(),
                      { data[eInt]:size(1) }))
    -- cut off output that goes after each other between overlap and overlap + step time step
    local modelOutput = self.model:forward(input, sizes)
    -- copy the first overlap - begining of the whole sequence
    for i = 0, overlap - 1 do
        output[{ { i + 1 } }]:copy(modelOutput[{ { i * iSeqs + 1 } }])
    end
    local linOutput = modelOutput[{ { overlap * iSeqs + 1, (overlap + step) * iSeqs } }]
    -- transform to regular sequence
    linOutput = linOutput:view(step, iSeqs, self.output_size)
    linOutput = linOutput:transpose(1, 2):reshape(iSeqs * step, self.output_size)
    output[{ { overlap + 1, overlap + linOutput:size(1) } }]:copy(linOutput)

    return output
end


function NeuralNetwork:test(dataset)
    assert(dataset.cols == self.input_size, "Dataset inputs does not match first layer size.")
    local g_error = 0
    local c_error = 0
    local b_count = 0
    local i_count = 0
    self.model:evaluate()
    local cLabels = true
    if torch.type(self.criterion) == self.CTC_CRITERION then
        cLabels = false
    end
    dataset:startBatchIteration(self.conf.sampling_type,
        self.conf.batch_size,
        false,
        self.conf.history,
        self.conf.split_sequences, cLabels)

    while true do
        local inputs, labels, sizes = dataset:nextBatch()
        if inputs == nil then
            break
        end
        self:_setTrainData(inputs, labels)
        b_count = b_count + 1
        local output = self.model:forward(self.inputs, sizes)
        g_error = g_error + self:_calculateError(output, self.labels)
        i_count = i_count + utils.sumTable(sizes)
        if self.criterion.forwardOnly then
            c_error = c_error + self.criterion:forwardOnly(output, self.labels,
                sizes, self.model.bSizes)
        else
            c_error = c_error + self.criterion:forward(output, self.labels,
                sizes, self.model.bSizes)
        end
    end
    collectgarbage()
    return c_error / b_count, g_error/i_count * 100
end


function NeuralNetwork:_setTrainData(inputs, labels)
    self.inputs:resize(inputs:size(1), inputs:size(2))
    self.inputs:copy(inputs)
    if torch.type(labels) ~= "table" then
        self.labels:resize(labels:size(1))
        self.labels:copy(labels)
    else
        self.labels = labels
    end
end


function NeuralNetwork:saveWeights(filename)
    print("Saving weights into: " .. filename)
    torch.save(filename, self.m_params:float())
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
    if self.conf.cuda then
        self.model = self.model:cuda()
    end
    self.m_params, self.m_grad_params = self.model:getParameters()
end


function NeuralNetwork:addLogSoftmax()
    print("Adding nn.LogSoftMax module to the model.")
    local sMax = nn.LogSoftMax()
    if self.conf.cuda then
        sMax = sMax:cuda()
    end
    self.model:add(sMax)
end

--eof
