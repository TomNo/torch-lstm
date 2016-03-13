require 'torch'
require 'optim'


-- overide standard rmsprop in the optim package
-- according to alex graves suggestion http://arxiv.org/abs/1308.0850

function optim.rmsprop(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) initialize mean square values and square gradient storage
    if not state.m then
      state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end

    -- (3) calculate new (leaky) mean squared values
    state.m:mul(alpha)
    state.m:addcmul(1.0-alpha, dfdx, dfdx)

    -- (4) perform update
    state.tmp:sqrt(state.m):add(epsilon)
    x:addcdiv(-lr, dfdx, state.tmp)

    -- return x*, f(x) before optimization
    return x, {fx}
end
