require 'torch' 
require 'cutorch'
require 'nn'
require 'json'

local NeuralNetwork = torch.class('NeuralNetwork')

NeuralNetwork.FEED_FORWARD = "FeedForward"
NeuralNetwork.MULTICLASS_CLASSIFICATION = "multiclass_classification"
NeuralNetowrk.SOFTMAX = "softmax"

function NeuralNetwork:__init(n_file)
  local f = assert(io.open(n_file, "r"))
  local net_desc = f:read("*all") -- this is strange might be better way how to read whole file
  f:close()
  -- description just as in the currennt
  self.network_file = n_file
  self.description = json.decode(net_desc)
  self.module = nn.Sequential()
  self:_createLayers()
end

function NeuralNetwork:_createLayers()
  if self.description["layers"] == nil then
    error("Missing layers section in " .. self.network_file .. ".")
  end
  
  for index, layer in self.description["layers"] do
    if index ~= #self.description["layers"] then
      if layer["name"] == nil or layer["bias"] == nil or layer["size"] == nil or layer["type"] == nil then
        error("Layer number: " .. index " is missing required attribute.")
      end
      self:addLayer(self.description["layers"][index], self.description["layers"][index+1])
    else -- last layer is objective function
      if layer["name"] == nil  or layer["size"] == nil or layer["type"] == nil then
        error("Layer number: " .. index " is missing required attribute.")
      end
      if layer["type"] == "MULTICLASS_CLASSIFICATION" then
        self.criterion = nn.ClassNLLCriterion()
      else
        error("Unknown objective function " .. layer["type"] .. ".")
      end      
    end
  end
end

function NeuralNetwork:_addLayer(layer, n_layer)
  if layer.type == NeuralNetwork.FEED_FORWARD then
    self.module:add(nn.Add(layer.bias))
    self.module:add(nn.Linear(layer.size, n_layer.size))
  else if layer.type == NeuralNetwork.SOFTMAX then
    self.module:add(nn.Add(layer.bias))
    self.module:add(nn.LogSoftMax())
  else
    error("Unknown layer type: " .. layer.type ".")
  end
end




