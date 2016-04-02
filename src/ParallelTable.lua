require 'torch'
require 'nn'

-- it is necessary to override ParallelTable because gradInput cannot be
-- used eficiently right now in official distribution, ideally there should
-- be global storage from witch could parallel module plug out tensors


local function backward(self, input, gradOutput, scale)
    for i, module in ipairs(self.modules) do
        module:backward(input[i], gradOutput[i], scale)
        if torch.isTensor(module.gradInput) then
            if self.gradInput[i] then
                self.gradInput[i]:resizeAs(module.gradInput)
                self.gradInput[i]:copy(module.gradInput)
            else
                self.gradInput[i] = module.gradInput:clone()
            end
        else -- means we have another table
            if not self.gradInput[i] or torch.type(self.gradInput[i]) ~= "table" then
                self.gradInput[i] = {}
            end
            for index, val in ipairs(module.gradInput) do
                if not torch.isTensor(val) then
                    error("Maximal level of depth is 2.")
                end
                if not self.gradInput[i][index] then
                    self.gradInput[i][index] = val:clone()
                else
                    self.gradInput[i][index]:resizeAs(val)
                    self.gradInput[i][index]:copy(val)
                end
            end
        end
    end
    return self.gradInput
end


nn.ParallelTable.backward = backward


--eof
