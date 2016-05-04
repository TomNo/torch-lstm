require 'torch'
require 'optim'


-- overide standard rmsprop in the optim package
-- and mix it up with the nesterov momentum
-- http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf


function optim.rmsprop(opfunc, x, config, state)
    config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-4
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8
    local momentum = config.momentum or 0.9
    local nesterov = config.nesterov_momentum or false

    -- if nesterov do the update first
    if state.update and nesterov then
        x:add(config.momentum, state.update)
    end

    local fx, dfdx = opfunc(x)
    -- init
    if not state.m then
        state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        --cache
        state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        -- previous update
        if momentum ~= 0 then
            state.update = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        end
    end

    -- calculate leaky squared values for gradients
    state.m:mul(alpha)
    state.m:addcmul(1.0 - alpha, dfdx, dfdx)

    state.tmp:sqrt(state.m):add(epsilon)
    if momentum ~= 0 then
        state.update:mul(momentum):addcdiv(-lr, dfdx, state.tmp)
        if nesterov then
            -- only correction update
            x:addcdiv(-lr, dfdx, state.tmp)
        else -- classical momentum
            x:add(state.update)
        end
    else
        x:addcdiv(-lr, dfdx, state.tmp)
    end

    -- return x*, f(x) before optimization
    return x, { fx }
end


-- according to alex graves suggestion http://arxiv.org/abs/1308.0850
-- adding gradient everage

function optim.grmsprop(opfunc, x, config, state)
    config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-4
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8
    local momentum = config.momentum or 0.9
    local nesterov = config.nesterov_momentum or false

    -- if nesterov do the update first
    if state.update and nesterov then
        x:add(config.momentum, state.update)
    end

    local fx, dfdx = opfunc(x)
    -- init
    if not state.m then
        state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        --cache
        state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        -- previous update
        if momentum ~= 0 then
            state.update = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        end
    end

    -- calculate leaky squared values for gradients
    state.m:mul(alpha)
    state.m:addcmul(1.0 - alpha, dfdx, dfdx)

    -- calculate leaky average of gradients
    state.g:mul(alpha)
    state.g:add(1.0 - alpha, dfdx)


    state.tmp:cmul(state.g, state.g):mul(-1):add(state.m):sqrt():add(epsilon)
    if momentum ~= 0 then
        state.update:mul(momentum):addcdiv(-lr, dfdx, state.tmp)
        if nesterov then
            -- only correction update
            x:addcdiv(-lr, dfdx, state.tmp)
        else -- classical momentum
            x:add(state.update)
        end
    else
        x:addcdiv(-lr, dfdx, state.tmp)
    end

    -- return x*, f(x) before optimization
    return x, { fx }
end
