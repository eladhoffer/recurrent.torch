recurrent.utils = {}

local function recursiveCopy(t1, t2)
    local t1, t2 = nn.utils.recursiveResizeAs(t1,t2)
    if torch.type(t2) == 'table' then
        t1 = (torch.type(t1) == 'table') and t1 or {t1}
        for key,_ in pairs(t2) do
            t1[key], t2[key] = recursiveCopy(t1[key], t2[key])
        end
    elseif torch.isTensor(t2) then
        t1 = torch.isTensor(t1) and t1:typeAs(t2) or t2.new()
        t1:resizeAs(t2)
        t1:copy(t2)
    else
        error("expecting nested tensors or tables. Got "..
        torch.type(t1).." and "..torch.type(t2).." instead")
    end
    return t1, t2
end

local function recursiveAdd(t1,t2)
    local t1, t2 = nn.utils.recursiveResizeAs(t1,t2)
    return nn.utils.recursiveAdd(t1,t2)
end

local function recursiveBatchExpand(t1, t2, batchSize)
    if torch.type(t2) == 'table' then
        t1 = (torch.type(t1) == 'table') and t1 or {t1}
        for key,_ in pairs(t2) do
            t1[key], t2[key] = recursiveBatchExpand(t1[key], t2[key], batchSize)
        end
    elseif torch.isTensor(t2) then
        t1 = torch.isTensor(t1) and t1 or t2.new()
        local sz = t2:size()
        sz[1] = batchSize
        t1:resize(sz)
        t1:copy(t2:expandAs(t1))
    else
        error("expecting nested tensors or tables. Got "..
        torch.type(t1).." and "..torch.type(t2).." instead")
    end
    return t1, t2
end


local function recursiveBatchResize(t, batchSize)
  if torch.type(t) == 'table' then
    for key,_ in pairs(t) do
      t[key] = recursiveBatchResize(t[key], batchSize)
    end
  elseif torch.isTensor(t) then
    if t:dim()> 0 then
      local sz = t:size()
      if sz[1] ~= batchSize then
        local init = sz[1] < batchSize
        sz[1] = batchSize
        t:resize(sz)
        if init then
          t:zero() --initialize values to zero
        end
      end
    end
  else
    error("expecting nested tensors or tables. Got "..
    torch.type(t).." instead")
  end
  return t
end

local function batchSize(x)
  if torch.isTensor(x) then
    return x:size(1)
  else
    return batchSize(x[1])
  end
end

recurrent.utils.recursiveBatchResize = recursiveBatchResize
recurrent.utils.recursiveBatchExpand = recursiveBatchExpand
recurrent.utils.recursiveCopy = recursiveCopy
recurrent.utils.recursiveAdd = recursiveAdd
recurrent.utils.batchSize = batchSize
