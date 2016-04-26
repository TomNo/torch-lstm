utils = {}

function utils.date()
    return os.date("_%H_%M_%d_%m_%y")
end


function utils.sumTable(tb)
    local sum = 0
    for k,v in pairs(tb) do
        sum = sum + v
    end

    return sum
end


function utils.getBatchSizes(sizes)
    local bSizes = {}
    for i=1, sizes[1] do
        local bSize = 0
        for y=1, #sizes do
            if sizes[y] >= i then
                bSize = bSize + 1
            else
                break
            end
        end
        table.insert(bSizes, bSize)
    end
    return bSizes
end


function utils.isNumber(num)
    return num > -math.huge and num < math.huge
end


function utils.removeNonNumbers(x)
    for i=#x, 1, -1 do
        if not utils.isNumber(x[i]) then
            table.remove(x, i)
        end
    end
    return x
end