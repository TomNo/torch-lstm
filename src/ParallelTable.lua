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
            self.gradInput[i] = module.gradInput
        end
    end
    return self.gradInput
end


nn.ParallelTable.backward = backward


--eof
