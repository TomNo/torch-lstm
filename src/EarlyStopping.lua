require 'torch'


--[[
-- EarlyStopping class represent the early stopping technique,
-- the improvement on cross validation error must be greater than 1%,
-- in order to be counted as improvement at all
 ]]
local EarlyStopping = torch.class('EarlyStopping')


EarlyStopping.MIN_DIFFERENCE_RATE = 0.01


function EarlyStopping:__init(history)
    self.history = history
    self.lError = math.huge
    self.bWeights = nil
    self.noBest = 1
end


function EarlyStopping:getBestWeights()
    return self.bWeights
end


function EarlyStopping:validate(net, dataset)
    local cv_c_error = net:test(dataset)
    local pError = self.lError
    print(string.format("Loss on cv set is: %.4f", cv_c_error))
    if cv_c_error < self.lError or not self.bWeights then
        if self.bWeights then
            self.bWeights:copy(net.m_params)
        else
            self.bWeights = net.m_params:float()
        end
        self.lError = cv_c_error
    end
    local mImpr = pError * self.MIN_DIFFERENCE_RATE
    local impr = pError - cv_c_error
    if cv_c_error >= pError or impr < mImpr then
        self.noBest = self.noBest + 1
    else
        self.noBest = 1
    end

    if self.noBest > self.history then
        return false
    end
    return true
end


return EarlyStopping
--eof
