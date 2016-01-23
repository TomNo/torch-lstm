require 'torch'
require 'nn'
require 'json'
require 'lstm'
require 'blstm'
require 'optim'

--TODO bptt?
--TODO gradiend cliping
-- TODO procesing sequences by uterrances
-- TODO ctc https://github.com/fchollet/keras/issues/383

torch.setdefaulttensortype('torch.FloatTensor')

-- Firt try to conver to number, then bool otherwise string is returned
-- TODO this can be pretty messy if something goes wrong -> terrible debugging
function parseConfigOption(val)
  local num_result = tonumber(val)
  if num_result ~= nil then
    return num_result
  elseif val == "true" then
    return true
  elseif val == "false" then
    return false
  end
  return val
end

CLIP_MIN = - 1.0
CLIP_MAX = 1.0

function grad_clip(element)
  if element > CLIP_MAX then
    return CLIP_MAX
  elseif element < CLIP_MIN then
    return CLIP_MIN
  else
    return element
  end
end 

local EarlyStopping = torch.class('EarlyStopping')

function EarlyStopping:__init(history)
  self.history = history
  self.w_hist = {}
  self.e_hist = {}
  for i=1, history do
    table.insert(self.w_hist, 0)
    table.insert(self.e_hist, math.huge)
  end
end

function EarlyStopping:getBestWeights()
  local b_error = math.huge
  local b_index = -1
  for i=1,self.history do
    if b_error > self.e_hist[i] then
      b_error = self.e_hist[i]
      b_index = i
    end
  end
  return self.w_hist[b_index]
end

-- attempts to insert new item
-- if possible returns true, false otherwise - learning should stop
function EarlyStopping:_insert(err, weights)
  local result = false
  for i=1, self.history do
    if err < self.e_hist[i] then
      self.e_hist[i] = err
      self.w_hist[i] = weights:clone()
      result = true
      break
    end
  end
  return result
end

function EarlyStopping:validate(net, dataset)
  local cv_g_error,  cv_c_error= net:test(dataset)
  print(string.format("Error on cv set set is: %.2f%% and loss is: %.4f",
   cv_g_error, cv_c_error))
  return self:_insert(cv_c_error, net.m_params)
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
    self[key] = value
  end
  self.desc = nil -- should contain json net topology description
  self.output_size = nil -- size of the last layer
  self.input_size = nil  -- size of the input layer
  self.conf = {} -- configuration regarding training
  self.m_params = nil -- model params
  self.m_grad_params = nil -- model gradients
  self.log = log  
  self.log:addTime('NeuralNetwork','%F %T')
end

function NeuralNetwork:init()
  print("Initializing neural network.")
  local f = assert(io.open(self.network_file, "r"),
   "Coult not open the network file: " .. self.network_file)
  local net_desc = f:read("*all") -- this is strange might be better way how to read whole file
  f:close()
  -- description just as in the currennt
  self.desc = json.decode(net_desc)
  self.output_size = self.desc.layers[#self.desc.layers-1].size
  self.input_size = self.desc.layers[1].size
  self.model = nn.Sequential()
  self:_parseConfig(self.config_file)
  if self.conf.cuda then
    -- load cuda if config says so
    if self.conf.cuda == 1 then
      local loadCuda = function()
                   require 'cutorch'
                   require 'cunn'
                 end
      local cudaEnabled = pcall(loadCuda)
      if not cudaEnabled then error("Could not load cuda.") end
      self.conf.cuda = true
    else
      self.conf.cuda = false
    end
  end
  self.e_stopping = EarlyStopping.new(self.conf.max_epochs_no_best)
  self:_createLayers()
  print("Model:")
  print(self.model)
  self.model:reset(self.conf.weights_uniform_max)
  if self.conf.optimizer == "rmsprop" then
    self.optim = optim.rmsprop
  elseif self.conf.optimizer == "adadelta" then
    self.optim = optim.adadelta
  else
    self.optim = optim.sgd
  end 
  self.m_params, self.m_grad_params = self.model:getParameters()
end

-- Parse configuration file, every line consist of key = value
-- save the config into to the self.conf
function NeuralNetwork:_parseConfig(conf_filename)
  local f = assert(io.open(conf_filename, "r"),
   "Could not open the configuration file: " .. conf_filename)
  local lines = f:lines()
  for line in lines do
    line = line:gsub("%s*", "")
    local result = line:split("=")
    self.conf[result[1]] = parseConfigOption(result[2])
  end
  f:close()
end

-- TODO input layer needs to be resolved
function NeuralNetwork:_createLayers()
  if self.desc.layers == nil then
    error("Missing layers section in " .. self.network_file .. ".")
  end

  for index, layer in ipairs(self.desc.layers) do
    if index > 1 and index ~= #self.desc.layers then
      if layer.name == nil or layer.bias == nil or layer.size == nil or layer.type == nil then
        error("Layer number: " .. index .. " is missing required attribute.")
      end
      self:_addLayer(self.desc.layers[index], self.desc.layers[index-1])
    elseif index  == #self.desc.layers then-- last layer is objective function
      if layer.name == nil  or layer.size == nil or layer.type == nil then
        error("Layer number: " .. index .. " is missing required attribute.")
    end
    if layer.type == NeuralNetwork.MULTICLASS_CLASSIFICATION then
      self.criterion = nn.ClassNLLCriterion()
      if self.conf.cuda then
        self.criterion = self.criterion:cuda()
      end
    else
      error("Unknown objective function " .. layer["type"] .. ".")
    end
    end
  end
  if self.conf.cuda then
    self.model = self.model:cuda()
  end
end

function NeuralNetwork:_addLayer(layer, p_layer)
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
    self.model:add(Lstm.new(p_layer.size, layer.size, self.conf.truncate_seq))-- layer.batch_normalization))
  elseif layer.type == NeuralNetwork.BLSTM then
    self.model:add(Blstm.new(p_layer.size, layer.size, self.conf.truncate_seq)) --layer.batch_normalization))
  elseif layer.type == NeuralNetwork.SOFTMAX then
    self.model:add(nn.Linear(p_layer.size, layer.size))
    self.model:add(nn.LogSoftMax())
  else
    error("Unknown layer type: " .. layer.type ".")
  end
  if layer.dropout and layer.dropout > 0 then
    self.model:add(nn.Dropout(layer.dropout))
  end
  if layer.batch_normalization then
    self.model:add(nn.BatchNormalization(layer.size))
  end
end

function NeuralNetwork:_calculateError(predictions, labels)
  local _, preds = predictions:max(2)
  return preds:typeAs(labels):ne(labels):sum()
end

--TODO add crossvalidation somehow??
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
    momentum = self.conf.momentum
--    learningRateDecay = self.conf.learning_rate_decay
  }
  local state = {}
  for epoch=1, self.conf.max_epochs do
    self.model:training()
    print('==> doing epoch ' .. epoch .. ' on training data.')
    local time = sys.clock()
    dataset:startIteration(self.conf.parallel_sequences,
                           self.conf.truncate_seq,
                           self.conf.shuffle_sequences,
                           true)
    local e_error = 0
    local e_predictions = 0
    local i_count = 0
    local grad_clips_acc = 0
    while true do
      self.inputs, self.labels = dataset:nextBatch()
      if self.inputs == nil then
        break
      end
      local feval = function(x)
        collectgarbage()
        -- get new parameters
        if x ~= self.m_params then
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
        grad_clips_acc = self.m_grad_params:eq(1):cat(self.m_grad_params:eq(-1)):sum() + grad_clips_acc
--        print("Max gradient: " .. self.m_grad_params:max())
--        print("Min gradient: " .. self.m_grad_params:min())
--        print("Average gradient: " .. self.m_grad_params:sum() / self.m_grad_params:nElement())
        return err, self.m_grad_params
      end -- feval
      self.optim(feval, self.m_params, opt_params, state)
    end -- mini batch
    collectgarbage()
    print("Epoch has taken " .. sys.clock() - time .. " seconds.")
    print("Gradient clippings occured " .. grad_clips_acc)
    grad_clips_acc = 0
    print(string.format("Error rate on training set is: %.2f%% and loss is: %.4f",
      e_predictions/i_count * 100, e_error))
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

function NeuralNetwork:forward(dataset)
  assert(dataset.cols == self.input_size,
   "Dataset input does not match first layer size.")
  self.model:evaluate()
  local outputs = torch.Tensor(dataset.rows, self.output_size)
  for i=1, dataset.rows, self.conf.parallel_sequences do
    local rows = self:_setActualBatchSize(i, dataset)
    self.inputs:copy(dataset.features[{{i, i+rows-1}, {}}])
    local labels = self.model:forward(self.inputs)
    outputs[{{i, i+rows - 1},{}}]:copy(labels) -- TODO there could also be just = labels:float()
  end
  collectgarbage()
  return outputs
end

-- calculates actual minibatch size and resize self.inputs and self.labels
-- return mini batch size
function NeuralNetwork:_setActualBatchSize(i, ds)
  local rows = math.min(i+self.conf.parallel_sequences, ds.rows + 1) - i
  if self.inputs:size(1) ~= rows then
    self.inputs:resize(rows, self.input_size)
  end
  if ds.labels and self.labels:size(1) ~= rows then
    self.labels:resize(rows)
  end
  return rows
end

function NeuralNetwork:test(dataset)
  assert(dataset.cols == self.input_size, "Dataset inputs does not match first layer size.")
  local g_error = 0
  local c_error = 0
  local i_count = 0
  self.model:evaluate()
--  dataset:startIteration(self.conf.parallel_sequences,
--                         self.conf.truncate_seq)

--  dataset:startSeqIteration()
  dataset:startRealBatch(self.conf.parallel_sequences, self.conf.truncate_seq, false, false)
  while true do
    self.inputs, self.labels = dataset:nextRealBatch()
    if self.inputs == nil then
      break
    end
    local output = self.model:forward(self.inputs)
    i_count = i_count + output:size(1)
    c_error = c_error + self.criterion(output, self.labels)
    g_error = g_error + self:_calculateError(output, self.labels)
  end
-- TODO this should be more likely a forward code
--  while true do
--    self.inputs, self.labels = dataset:getSeq()
--    if self.inputs == nil then
--      break
--    end
--    o_labels:resize(self.inputs:size(1), self.output_size)
--    local it = 1
--    while true do
--      local e_int = it + self.conf.truncate_seq - 1
--      local diff = e_int - self.inputs:size(1)
--      if  diff >= self.conf.truncate_seq then
--        break -- next mini sequences is entirely behind current seq maximum
--      elseif diff > 0 then
--        e_int = self.inputs:size(1) -- make it end at the whole sequence end
--      end  
--        
--      local interval = {{it, e_int},{}}
--      o_labels[interval] = self.model:forward(self.inputs[interval])
--          
--      it = it + self.conf.truncate_seq
--    end
--    c_error = c_error + self.criterion(o_labels, self.labels)
--    for c=1, self.labels:size(1) do
--      local _, l_max =  o_labels[c]:max(1)
--      if l_max[1] ~= self.labels[c] then
--        g_error = g_error + 1
--      end    
--    end 
--  end
  -- TODO refactor
--  while true do
--    self.inputs, self.labels = dataset:nextBatch()
--    if self.inputs == nil then
--      break
--    end
--    local o_labels = self.model:forward(self.inputs)
--    c_error = c_error + self.criterion(o_labels, self.labels)
--    for c=1, o_labels:size(1) do
--      local _, l_max =  o_labels[c]:max(1)
--      if l_max[1] ~= self.labels[c] then
--        g_error = g_error + 1
--      end    
--    end
--  end
  collectgarbage()
  return (g_error / i_count) * 100, c_error
end

function NeuralNetwork:saveModel(filename)
  torch.save(filename, self.model)
end

function NeuralNetwork:loadModel(filename)
  self.model = torch.load(filename)
end

