require 'nn'

-- sharing does not work for tables
local function share(self, mlp, ...)
    local arg = { ... }
    for i, v in ipairs(arg) do
        if self[v] ~= nil then
            if self[v].set then
                self[v]:set(mlp[v])
            end
            self.accUpdateGradParameters = self.sharedAccUpdateGradParameters
            mlp.accUpdateGradParameters = mlp.sharedAccUpdateGradParameters
        end
    end
    return self
end


nn.Module.share = share


--eof
