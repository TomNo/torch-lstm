require 'torch'
require 'nn'
require 'json'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

loadCuda = function()
  require 'cutorch'
  require 'cunn'
end
--
cudaEnabled = pcall(loadCuda)

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
  self:_createLayers()
  self:_parseConfig(self.config_file)
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
      if cudaEnabled then
        self.criterion = self.criterion:cuda()
      end
    else
      error("Unknown objective function " .. layer["type"] .. ".")
    end
    end
  end
  if cudaEnabled then
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
    error("Lstm cell is not supported yet.")
  elseif layer.type == NeuralNetwork.BLSTM then
    error("Blstm cell is not supported yet.")
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
  if cudaEnabled then
    print("Training on GPU.")
  else
    print("Training on CPU.")
  end
  assert(dataset[1][1]:size(1) == self.input_size,
   "Dataset input does not fit first layer size.")
  local opt_params = {
    learningRate = self.conf.learning_rate,
    weightDecay = self.conf.weight_decay,
    momentum = self.conf.momentum,
    learningRateDecay = self.conf.learning_rate_decay
  }
  self.model:training() -- not sure that it is necessary
  for epoch=1, self.conf.max_epochs do
    print('==> doing epoch ' .. epoch .. ' on training data.')
    local time = sys.clock()
    local shuffle
    if self.conf.shuffle_sequences == true then
      shuffle = torch.randperm(dataset:size())
    else
      shuffle = torch.range(1, dataset:size())
    end

    for i=1, dataset:size(), self.conf.parallel_sequences do
      local inputs = torch.Tensor(self.conf.parallel_sequences, dataset[1][1]:nElement())
      local labels = torch.Tensor(self.conf.parallel_sequences, dataset[1][2]:nElement())
      local index = 1
      for y=i, math.min(i+self.conf.parallel_sequences - 1, dataset:size()) do
        inputs[index] = dataset[shuffle[y]][1]
        labels[index] = dataset[shuffle[y]][2]
        index = index + 1
      end
      labels = labels:squeeze()
      if cudaEnabled then
        labels = labels:cuda()
        inputs = inputs:cuda()
      end

      local feval = function(x)
        collectgarbage()
        -- get new parameters
        if x ~= self.m_params then
          self.m_params:copy(x)
        end

        -- reset gradients
        self.m_grad_params:zero()
        local outputs = self.model:forward(inputs)
        local err = self.criterion:forward(outputs, labels)
        self.model:backward(inputs, self.criterion:backward(outputs, labels))

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
      if cv_dataset then
        local cv_g_error,  cv_c_error= self:test(cv_dataset)
        print("Error on cv set set is: " .. cv_g_error .. "% " .. cv_c_error)
      end
    end
  end -- epoch
end

function NeuralNetwork:forward(dataset)
  assert(dataset[1][1]:size(1) == self.input_size,
   "Dataset input does not match first layer size.")
  local outputs = torch.Tensor(dataset:size(), self.output_size)
  for i=1, dataset:size(), self.conf.parallel_sequences do
    local rows = math.min(i+self.conf.parallel_sequences, dataset:size() + 1) - i
    local inputs = torch.Tensor(rows, dataset[1][1]:nElement())
    local index = 1
    for y=i, math.min(i+self.conf.parallel_sequences-1, dataset:size()) do
      inputs[index] = dataset[y][1]
      index = index + 1
    end
    if cudaEnabled then
      inputs = inputs:cuda()
    end
    local labels = self.model:forward(inputs)
    outputs[{{i, i+rows - 1},{}}]:copy(labels) -- TODO there could also be just = labels:float()
  end
  collectgarbage()
  return outputs
end

function NeuralNetwork:test(dataset)
  assert(dataset[1][1]:size(1) == self.input_size, "Dataset inputs does not match first layer size.")
  local g_error = 0
  local c_error = 0
  for i=1, dataset:size(), self.conf.parallel_sequences do
    local rows = math.min(i+self.conf.parallel_sequences, dataset:size() + 1) - i
    local inputs = torch.Tensor(rows, self.input_size)
    local targets = torch.Tensor(rows, dataset[1][2]:size(1))
    local index = 1
    for y=i, math.min(i+self.conf.parallel_sequences-1, dataset:size()) do
      inputs[index] = dataset[y][1]
      targets[index] = dataset[y][2]
      index = index + 1
    end
    targets = targets:squeeze()
    if cudaEnabled then
      inputs = inputs:cuda()
      targets = targets:cuda()
    end
    local labels = self.model:forward(inputs)
    c_error = c_error + self.criterion(labels, targets)
    for c=1, labels:size(1) do
      local _, l_max =  labels[c]:max(1)
      if l_max[1] ~= targets[c] then
        g_error = g_error + 1
      end    
    end
  end
  collectgarbage()
  return (g_error / dataset:size()) * 100, c_error
end

function NeuralNetwork:saveModel(filename)
  torch.save(filename, self.model)
end

function NeuralNetwork:loadModel(filename)
  self.model = torch.load(filename)
end

