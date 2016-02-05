require 'torch'


local Configuration = torch.class("Configuration")


function Configuration:__init(filename)
    self:_parse(filename)
end


-- Firt try to conver to number, then bool otherwise string is returned
-- TODO this can be pretty messy if something goes wrong -> terrible debugging
function Configuration:_parseOption(val)
    local num_result = tonumber(val)
    if num_result ~= nil then
        return num_result
    elseif val == "true" then
        return true
    elseif val == "false" then
        return false
    end
    return val
end

-- Parse configuration file, every line consist of key = value
-- save the config into to the self.conf
function Configuration:_parse(filename)
    local f = assert(io.open(filename, "r"),
        "Could not open the configuration file: " .. filename)
    local lines = f:lines()
    for line in lines do
        line = line:gsub("%s*", "")
        local result = line:split("=")
        if self[result[1]] then
            error(string.format("Attribute %s, is already present in the configuration.", result[1]))
        else
            self[result[1]] = self:_parseOption(result[2])
        end
    end
    f:close()
end


return Configuration