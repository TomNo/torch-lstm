require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Swaps the position of the blank symbol in the input file.')
cmd:option('--input_file', 'input.hdf5', 'Input file')
cmd:option('--output_file', 'output_swapped.hdf5', 'Output file with swapped blank')

params = cmd:parse(arg)

pr =  {4, 4}
local iFile = hdf5.open(params.input_file)
local iData = iFile:read():all()
for _, val in pairs(iData) do
    local size = val["data"]:size(2)
    local a1 = val["data"][{{},{1, 1}}]:clone()
    local a2 = val["data"][{{}, pr}]:clone()
  val["data"][{{}, {1,1}}] = a2
  val["data"][{{}, pr}] = a1
end

local oFile = hdf5.open(params.output_file, "w")
oFile:write("/", iData)
oFile:close()
