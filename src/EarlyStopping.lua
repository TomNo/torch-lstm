require 'torch'

--TODO it does not work as expected


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


return EarlyStopping