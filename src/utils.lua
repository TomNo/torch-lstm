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



-- modified https://gist.github.com/Nayruden/427389
function utils.leven(s, t)
    local s_len, t_len = #s, #t

    local min = math.min
    local num_columns = t_len + 1

    local d = {}

    for i=0, s_len do
        d[ i * num_columns ] = i
    end
    for j=0, t_len do
        d[ j ] = j
    end

    for i=1, s_len do
        local i_pos = i * num_columns
        for j=1, t_len do
            local add_cost = (s[ i ] ~= t[ j ] and 1 or 0)
            local val = min(
                d[ i_pos - num_columns + j ] + 1,
                d[ i_pos + j - 1 ] + 1,
                d[ i_pos - num_columns + j - 1 ] + add_cost
            )
            d[ i_pos + j ] = val

            if i > 1 and j > 1 and s[ i ] == t[ j - 1 ] and s[ i - 1 ] == t[ j ] then
                d[ i_pos + j ] = min(
                    val,
                    d[ i_pos - num_columns - num_columns + j - 2 ] + add_cost
                )
            end

        end
    end
    return d[#d]
end

--eof
