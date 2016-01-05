require 'torch'
require 'nn'
require 'json'
require 'lstm'
require 'blstm'
require 'optim'

--TODO bptt?
--TODO gradiend cliping

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

local NeuralNetwork = torch.class('NeuralNetwork')

NeuralNetwork.FEED_FORWARD_TANH = "feedforward_tanh"
NeuralNetwork.FEED_FORWARD_LOGISTIC = "feedforward_logistic"
NeuralNetwork.MULTICLASS_CLASSIFICATION = "multiclass_classification"
NeuralNetwork.FEED_FORWARD_RELU = "feedforward_relu"
NeuralNetwork.SOFTMAX = "softmax"
NeuralNetwork.INPUT = "input"
NeuralNetwork.LSTM = "lstm"
NeuralNetwork.BLSTM = "blstm"

function NeuralNetwork:__init(params, log)
  for key, value in pairs(params) do
    self[key] = value
  end
  self.desc = nil -- should contain json net topology description
  self.output_size = nil -- size of the last layer
  self.input_size = nil  -- size of the input layer
  self.conf = {} -- configuration regarding training
  self.m_params = nil -- model
  self.m_grad_params = nil -- model
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
  self:_createLayers()
  local allocator = torch.Tensor
  if self.conf.cuda then allocator = torch.CudaTensor end
  self.inputs = allocator(self.conf.parallel_sequences, self.input_size)
  self.labels = allocator(self.conf.parallel_sequences)
end

-- Parse configuration file, every line consist of key = value
-- save the config into to the self.conf
-- TODO network description file name is contained in config.cfg
function NeuralNetwork:_parseConfig(conf_filename)
  local f = assert(io.open(conf_filename, "r"),
   "Could not open the configuration file: " .. conf_filename)
  local lines = f:lines()
  for line in lines do
    -- :D lua support for string is only via regexs, how sad :D
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
  local params, g_p = self.model:getParameters()
  self.m_params = params
  self.m_grad_params = g_p
end

function NeuralNetwork:_addLayer(layer, p_layer)
  if layer.type == NeuralNetwork.INPUT then
    return -- just for backward compatibility
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
    self.model:add(Lstm.new(p_layer.size, layer.size, self.conf.truncate_seq))
  elseif layer.type == NeuralNetwork.BLSTM then
    self.model:add(Blstm.new(p_layer.size, layer.size, self.conf.truncate_seq))
  elseif layer.type == NeuralNetwork.SOFTMAX then
    --    self.model:add(nn.Add(layer.bias))
    self.model:add(nn.Linear(p_layer.size, layer.size))
    self.model:add(nn.LogSoftMax())
  else
    error("Unknown layer type: " .. layer.type ".")
  end
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
    momentum = self.conf.momentum,
    learningRateDecay = self.conf.learning_rate_decay
  }
  self.model:training()
  for epoch=1, self.conf.max_epochs do
    print('==> doing epoch ' .. epoch .. ' on training data.')
    local time = sys.clock()
    local shuffle = nil
    if self.conf.shuffle_sequences == true then
      shuffle = torch.randperm(dataset.rows)
    else
      shuffle = torch.range(1, dataset.rows)
    end
    dataset:startBatchIteration(self.conf.parallel_sequences,
                                self.conf.truncate_seq)
    for i=1, dataset.rows, (self.conf.parallel_sequences * self.conf.truncate_seq) do
--      local b_size = self:_setActualBatchSize(i, dataset)
--      local index = 1
--      -- TODO cannot be copied because of shuffle
--      for y=i, i + b_size -1 do
--        self.inputs[index] = dataset.features[shuffle[y]]
--        self.labels[index] = dataset.labels[shuffle[y]]
--        index = index + 1
--      end
      self.inputs, self.labels = dataset:getBatch()
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
        self.model:backward(self.inputs, self.criterion:backward(outputs, self.labels))
        -- normalize gradients and error
--        self.m_grad_params:div(inputs:nElement())
--        err = err/inputs:nElement()

        -- return f and df/dX
        return err, self.m_grad_params
      end -- feval
      optim.sgd(feval, self.m_params, opt_params)
    end -- mini batch
    collectgarbage()
    print("Epoch has taken " .. sys.clock() - time .. " seconds.")
    if not self.conf.validate_every or epoch % self.conf.validate_every == 0 then
      local g_error,  c_error= self:test(dataset)
      print("Error on training set is: " .. g_error .. "% " .. c_error)
--      if cv_dataset then
--        local cv_g_error,  cv_c_error= self:test(cv_dataset)
--        print("Error on cv set set is: " .. cv_g_error .. "% " .. cv_c_error)
--      end
    end
  end -- epoch
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
  dataset:startBatchIteration(self.conf.parallel_sequences,
                              self.conf.truncate_seq)
  -- TODO refactor
  for i=1, dataset.rows, (self.conf.parallel_sequences * self.conf.truncate_seq) do
--    local rows = self:_setActualBatchSize(i, dataset)
--    self.inputs:copy(dataset.features[{{i, i+rows-1}, {}}])
--    self.labels:copy(dataset.labels[{{i, i+rows-1}}])
    self.inputs, self.labels = dataset:getBatch()
    if self.inputs == nil then
      break
    end
    local o_labels = self.model:forward(self.inputs)
    c_error = c_error + self.criterion(o_labels, self.labels)
    for c=1, o_labels:size(1) do
      local _, l_max =  o_labels[c]:max(1)
      if l_max[1] ~= self.labels[c] then
        g_error = g_error + 1
      end    
    end
  end
  collectgarbage()
  return (g_error / dataset.rows) * 100, c_error
end

function NeuralNetwork:saveModel(filename)
  torch.save(filename, self.model)
end

function NeuralNetwork:loadModel(filename)
  self.model = torch.load(filename)
end

