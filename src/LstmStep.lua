require 'torch'
require 'nn'
require 'Step'
require 'LinearScale'
require 'AddLinear'


-- INFO no batch norm because it performs worse
-- http://arxiv.org/pdf/1510.01378.pdf


local LstmStep = torch.class('nn.LstmStep', 'nn.Step')

--TODO rename variables --> camelCase
--TODO make it more elegent
function LstmStep:__init(layerSize)
    nn.Step.__init(self, layerSize)
    self.inputSize = 4 * layerSize
    self.layerSize = layerSize
    self.cellDeltas = torch.Tensor()
    -- all output activations
    self.oActs = nn.AddLinear(layerSize, 4 * layerSize)
    self.iPeeps = nn.LinearScale(layerSize)
    self.fPeeps = nn.LinearScale(layerSize)
    self.oPeeps = nn.LinearScale(layerSize)
    self.ifSigmoid = nn.Sigmoid()
    self.icTanh = nn.Tanh()
    self.iScale = nn.CMulTable()
    self.fScale = nn.CMulTable()
    self.cState = nn.CAddTable(true)
    self.oSigmoid = nn.Sigmoid()
    self.ocTanh = nn.Tanh()
    self.oScale = nn.CMulTable()

    self.iInt = {{}, {1, self.layerSize} }
    self.fInt = {{}, {self.layerSize + 1, 2 * self.layerSize}}
    self.cInt = {{}, {2*self.layerSize + 1, 3*self.layerSize} }
    self.oInt = {{}, {3*self.layerSize + 1, 4* self.layerSize} }
    self.ifInt = {{}, {1, 2*self.layerSize} }

    local mNames = {"oActs", "iPeeps", "fPeeps", "oPeeps", "ifSigmoid",
        "icTanh", "iScale", "fScale", "cState", "oSigmoid", "ocTanh", "oScale"}
    for i=1, #mNames do
        self:add(self[mNames[i]])
    end

    self.gradInput = torch.Tensor()
end


function LstmStep:updateOutput(input)
    self.input, self.pOutput, self.pCellOutput = self:currentInput(input)
    self.oActs:forward({input, self.pOutput})
    -- peepholes
    self.iPeeps:forward({self.oActs.output[self.iInt], self.pCellOutput})
    self.fPeeps:forward({self.oActs.output[self.fInt], self.pCellOutput})
    self.ifSigmoid:forward(self.oActs.output[self.ifInt])

    self.icTanh:forward(self.oActs.output[self.cInt])
    self.iScale:forward({self.icTanh.output, self.ifSigmoid.output[self.iInt]})
    self.fScale:forward({self.pCellOutput, self.ifSigmoid.output[self.fInt]})

    self.cState:forward({self.iScale.output, self.fScale.output})

    self.oPeeps:forward({self.oActs.output[self.oInt], self.cState.output})
    self.oSigmoid:forward(self.oActs.output[self.oInt])
    self.ocTanh:forward(self.cState.output)
    self.oScale:forward({self.ocTanh.output, self.oSigmoid.output})
    self.output = self.oScale.output
    return self.output
end

function LstmStep:backward(input, gradOutput, scale)
    local function backward(obj, i, g)
        obj:backward(i, g, scale)
    end
    local rInt = {1, input:size(1) }
    self.oActs[1] = rInt
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    if self.nStep then
        local deltas = self.nStep():getOutputDeltas()
        local rInt = {{1, math.min(deltas:size(1), gradOutput:size(1))}}
        gradOutput[rInt]:add(deltas[rInt])
    end
    backward(self.oScale, {self.ocTanh.output, self.oSigmoid.output}, gradOutput)
    backward(self.ocTanh, self.cState.output, self.oScale.gradInput[1])
    if self.nStep then
        local cellDeltas = self.nStep():getCellDeltas()
        local rInt = {{1, math.min(cellDeltas:size(1), self.ocTanh.gradInput:size(1))}}
        self.ocTanh.gradInput[rInt]:add(cellDeltas[rInt])
    end
    self.oSigmoid.gradInput = self.gradInput[self.oInt]
    backward(self.oSigmoid, self.oActs.output[self.oInt], self.oScale.gradInput[2])
    backward(self.oPeeps, {self.oActs.output[self.oInt], self.cState.output[{rInt}]}, self.oSigmoid.gradInput)
    self.ocTanh.gradInput:add(self.oPeeps.gradInput)
    backward(self.cState, {self.iScale.output, self.fScale.output}, self.ocTanh.gradInput)
    backward(self.fScale, {self.pCellOutput, self.ifSigmoid.output[self.fInt]}, self.cState.gradInput[2])
    backward(self.iScale, {self.icTanh.output, self.ifSigmoid.output[self.iInt]}, self.cState.gradInput[1])
    self.icTanh.gradInput = self.gradInput[self.cInt]
    backward(self.icTanh, self.oActs.output[self.cInt], self.iScale.gradInput[1])
    self.ifSigmoid.gradInput = self.gradInput[self.ifInt]
    backward(self.ifSigmoid, self.oActs.output[self.ifInt], self.iScale.gradInput[2]:cat(self.fScale.gradInput[2]))
    backward(self.fPeeps, {self.oActs[self.fInt], self.pCellOutput}, self.ifSigmoid.gradInput[self.fInt])
    self.cellDeltas:resizeAs(self.fPeeps.gradInput)
    self.cellDeltas:copy(self.fPeeps.gradInput)
    backward(self.iPeeps, {self.oActs[self.iInt], self.pCellOutput}, self.ifSigmoid.gradInput[self.iInt])
    self.cellDeltas:add(self.iPeeps.gradInput)
    self.cellDeltas:add(self.fScale.gradInput[1])
    backward(self.oActs, {input, self.pOutput}, self.gradInput)
    return self.gradInput
end


function LstmStep:getGradInput()
    return self.gradInput
end


function LstmStep:getCellStates()
    return self.cState.output
end


function LstmStep:getCellDeltas()
    return self.cellDeltas
end


function LstmStep:getOutputDeltas()
    return self.oActs.gradInput[2]
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
    local rInt = {{1, input:size(1)} }
    if pOutput:size(1) < input:size(1) then
        local pSize = pOutput:size(1)
        pOutput:resize(input:size(1), pOutput:size(2))
        pCellStates:resize(input:size(1), pCellStates:size(2))
        pOutput[{{pSize + 1, input:size(1)}}]:zero()
        pCellStates[{{pSize + 1, input:size(1)}}]:zero()
    end

    return input, pOutput[rInt], pCellStates[rInt]
end

--eof
