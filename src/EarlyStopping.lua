require 'torch'


local EarlyStopping = torch.class('EarlyStopping')


function EarlyStopping:__init(history)
    self.history = history
    self.w_hist = {}
    self.e_hist = {}
    for i = 1, history do
        table.insert(self.w_hist, 0)
        table.insert(self.e_hist, math.huge)
    end
end


function EarlyStopping:getBestWeights()
    local b_error = math.huge
    local b_index = -1
    for i = 1, self.history do
        if b_error > self.e_hist[i] then
            b_error = self.e_hist[i]
            b_index = i
        end
    end
    return self.w_hist[b_index], b_index, b_error
end


function EarlyStopping:_insert(err, weights)
    local _, b_index, b_error = self:getBestWeights()
    -- insert new weights on the first position
    if err < b_error or b_index < self.history then
        table.insert(self.e_hist, 1, err)
        table.insert(self.w_hist, 1, weights:clone())
        self.e_hist[self.history + 1] = nil
        self.w_hist[self.history + 1] = nil
        return true
    else
        return false
    end
end


function EarlyStopping:validate(net, dataset)
    local cv_g_error, cv_c_error = net:test(dataset)
    print(string.format("Error on cv set set is: %.2f%% and loss is: %.4f",
        cv_g_error, cv_c_error))
    return self:_insert(cv_c_error, net.m_params)
end


return EarlyStopping