require 'torch'


local EarlyStopping = torch.class('EarlyStopping')


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
    print(string.format("Loss on cv set is: %.4f", cv_c_error))
    if cv_c_error < self.lError then
        if self.bWeights then
            self.bWeights:copy(net.m_params)
        else
            self.bWeights = net.m_params:float()
        end
        self.noBest = 1
        self.lError = cv_c_error
    else
        self.noBest = self.noBest + 1
    end
    if self.noBest > self.history then
        return false
    end
    return true
end


return EarlyStopping