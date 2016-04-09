require 'torch'
require 'nn'
require 'Step'
require 'LinearScale'
require 'Split'
require 'AddLinear'


-- INFO no batch norm because it performs worse
-- http://arxiv.org/pdf/1510.01378.pdf


local LstmStep = torch.class('nn.LstmStep', 'nn.Step')

--TODO rename variables --> camelCase
--TODO make it more elegent
function LstmStep:__init(layerSize)
    nn.Step.__init(self, layerSize)
    self.inputSize = 4 * layerSize
    self.cellStates = nil
    -- all output activations
    local o_acts = nn.AddLinear(layerSize, 4 * layerSize)
    -- set bias to 1 because of the forget gate activation
    o_acts.bias:fill(1)
    --  -- forget and input peepholes cell acts
    local fg_peep = nn.Sequential():add(nn.ConcatTable():add(nn.LinearScale(layerSize)):add(nn.LinearScale(layerSize))):add(nn.JoinTable(2))
    -- add forget and input peepholes
    local c_acts = nn.ConcatTable():add(fg_peep):add(nn.Identity())

    -- container for summed input and output activations
    -- that is divided in half
    local io_acts = nn.Sequential():add(o_acts):add(nn.Split(2))
--    io_acts:add(nn.ParallelTable():add(i_acts):add(o_acts))
--    io_acts:add(nn.CAddTable(true)):add(nn.Split(2))
    -- sum half of the activations with peepholes
    self:add(nn.ParallelTable():add(io_acts):add(c_acts))
    self:add(nn.FlattenTable())
    -- output of the model at this stage is <c_states + o_acts, i_acts + f_acts, peepholes acts, cell states>
    -- input and forget gate activation
    local items = nn.ConcatTable()
    items:add(nn.Sequential():add(nn.NarrowTable(2, 2)):add(nn.CAddTable(true)):add(nn.RegularSigmoid(true)):add(nn.Split(2)))
    items:add(nn.Sequential():add(nn.SelectTable(4)))
    --    --  -- divide rest activations between cell state and output gate
    items:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Split(2)))
    self:add(items)
    self:add(nn.FlattenTable())
    -- output of the model at this stage is <i_acts, f_acts, cell states, c_acts, o_acts>
    items = nn.ConcatTable()
    -- forward i_acts
    items:add(nn.SelectTable(1))
    -- apply squashing function to cell state
    items:add(nn.Sequential():add(nn.SelectTable(4)):add(nn.RegularTanh()))
    -- apply forgeting
    items:add(nn.Sequential():add(nn.NarrowTable(2, 2)):add(nn.CMulTable()))
    -- forward o_acts
    items:add(nn.SelectTable(5))
    -- output of the model at this stage is <i_acts, c_acts, f_acts, o_acts>
    self:add(items)
    items = nn.ConcatTable()
    -- scale cell state by input
    items:add(nn.Sequential():add(nn.NarrowTable(1, 2)):add(nn.CMulTable()))
    -- forward
    items:add(nn.Sequential():add(nn.SelectTable(3)))
    items:add(nn.Sequential():add(nn.SelectTable(4)))
    -- output of the model at this stage is <c_acts, f_acts, o_acts>
    -- add previous cell state
    self:add(items)
    local tmp = nn.ConcatTable()
    tmp:add(nn.Sequential():add(nn.NarrowTable(1, 2)):add(nn.CAddTable(true)))
    tmp:add(nn.Sequential():add(nn.SelectTable(3)))
    self.cellActs = tmp
    self:add(tmp)
    -- output of the model at this stage is <c_acts, o_acts>
    -- scale by peephole from the cell state to output gate and apply sigmoid to output gate,
    -- also apply squashing function to the cell states
    tmp = nn.ConcatTable()
    --  if self.b_norm then
    --    cell_acts:add(nn.Sequential():add(nn.SelectTable(1)):add(LinearNoBias.new(layerSize, layerSize)):add(nn.BatchNormalization(layerSize)))
    --  else
    tmp:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.LinearScale(layerSize)))
    --  end
    tmp:add(nn.SelectTable(2))
    tmp:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Tanh()))
    self:add(tmp) -- 8th module
    -- output of the model at this stage is <output_gate peephole act, o_acts, cell_acts>
    -- finalize the o_acts and apply sigmoid
    tmp = nn.ConcatTable():add(nn.Sequential():add(nn.NarrowTable(1, 2)):add(nn.CAddTable(true)):add(nn.Sigmoid(true)))
    -- just forward cell acts
    tmp:add(nn.SelectTable(3))
    -- result is <output>
    self:add(tmp)
    self:add(nn.CMulTable())
end


function LstmStep:backward(input, gradOutput, scale)
    scale = scale or 1
    local nGradOutput, nCellGradOutput
    if self.nStep then
        nGradOutput = self.nStep():getOutputDeltas()
        nCellGradOutput = self.nStep():getCellDeltas()
        gradOutput:add(nGradOutput)
    end

    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i = #self.modules - 1, 1, -1 do
        local previousModule = self.modules[i]
        -- adding cell deltas
        if currentModule == self.cellActs and nCellGradOutput then
            currentGradOutput[1]:add(nCellGradOutput)
        end
        currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
        currentModule.gradInput = currentGradOutput
        currentModule = previousModule
    end
    currentGradOutput = currentModule:backward(self:currentInput(input), currentGradOutput, scale)
    self.gradInput = currentGradOutput
    return currentGradOutput
end


function LstmStep:getGradInput()
    return self.gradInput[1][1]
end


function LstmStep:getCellStates()
    return self.cellActs.output[1]
end


function LstmStep:getCellDeltas()
    return self.gradInput[2]
end


function LstmStep:getOutputDeltas()
    return self.gradInput[1][2]
end


function LstmStep:currentInput(input)
    local pOutput, pCellStates
    if self.pStep then
        local pStep = self.pStep()
        pOutput = pStep.output
        pCellStates = pStep:getCellStates()
    else
        self:adaptZTensor()
        pOutput = self.zTensor:expand(input:size(1), self.layerSize)
        pCellStates = pOutput
    end
    return { { input, pOutput }, pCellStates }
end

--eof
