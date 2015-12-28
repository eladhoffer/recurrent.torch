local Recurrent, parent = torch.class('nn.RecurrentContainer', 'nn.Container')

local function recursiveCopy(t1, t2)
    if torch.type(t2) == 'table' then
        t1 = (torch.type(t1) == 'table') and t1 or {t1}
        for key,_ in pairs(t2) do
            t1[key], t2[key] = recursiveCopy(t1[key], t2[key])
        end
    elseif torch.isTensor(t2) then
        t1 = torch.isTensor(t1) and t1 or t2.new()
        t1:resizeAs(t2)
        t1:copy(t2)
    else
        error("expecting nested tensors or tables. Got "..
        torch.type(t1).." and "..torch.type(t2).." instead")
    end
    return t1, t2
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
            t[key] = recursiveBatchResize(t, batchSize, dims)
        end
    elseif torch.isTensor(t) then
        local sz = t:size()
        if sz[1] ~= batchSize then
            sz[1] = batchSize
            t:resize(sz)
            t:zero() --initialize values to zero
        end
    else
        error("expecting nested tensors or tables. Got "..
        torch.type(t).." instead")
    end
    return t
end


function Recurrent:__init(recurrentModule)
    parent.__init(self)
    local recurrentModule = recurrentModule or nn.Sequential()
    self.modules = {}
    self.state = torch.Tensor()
    self.gradState = torch.Tensor()
    self.initState = torch.Tensor()
    self.timeDimension = 2
    self.splitInput = nn.SplitTable(self.timeDimension)
    self.joinOutput = nn.JoinTable(self.timeDimension)
    self.currentIteration = 1
    self.seqMode = false

    self:add(recurrentModule)
end

function Recurrent:setMode(mode) --mode can be 'sequence' or 'single'
    assert(mode == 'sequence' or mode == 'single')
    self.seqMode = (mode == 'sequence')
    return self
end

function Recurrent:single()
    return self:setMode('single')
end

function Recurrent:sequence()
    return self:setMode('sequence')
end


function Recurrent:setIterations(iterations, clear)
    local start = 2
    if clear then
        self.modules = {self.modules[1]}
    else
        start = #self.modules + 1
    end
    for i=start, iterations do
        self.modules[i] = self.modules[1]:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
    end
end

function Recurrent:add(m)
    if torch.type(m) =='table' and (m.rnnModule) and (m.initState) then
        self:setState(m.initState)
        self.name = m.name
        m = m.rnnModule
    end
    if self.modules[1] then
        self.modules[1]:add(m)
    else
        self.modules[1] = m
    end
    self.gradInput = self.modules[1].gradInput
    self.output = self.modules[1].output
    self:setIterations(#self.modules, true)
    return self
end

function Recurrent:insert(m,i)
    self.modules[1]:insert(m,i)
    self:setIterations(#self.modules, true)
end

function Recurrent:remove(m,i)
    self.modules[1]:remove(m,i)
    self:setIterations(#self.modules, true)
end


function Recurrent:setState(state, batchSize)
    if batchSize then
        self.state = recursiveBatchExpand(self.state, state, batchSize)
    else
        self.state = recursiveCopy(self.state, state)
    end
end

function Recurrent:getState()
    return self.state
end

function Recurrent:resizeStateBatch(batchSize)
    self.state = recursiveBatchResize(self.state, batchSize)
end

function Recurrent:setGradState(gradState)
    self.gradState = recursiveCopy(self.gradState, gradState)
end

function Recurrent:getGradState()
    return self.gradState
end

function Recurrent:zeroGradState()
    self.gradState = nn.utils.recursiveResizeAs(self.gradState, self.state)
    nn.utils.recursiveFill(self.gradState, 0)
end

function Recurrent:zeroState()
    nn.utils.recursiveFill(self.state, 0)
    self:zeroGradState()
end

function Recurrent:forget(release)
    self.currentIteration = 1
    self:setIterations(1, release)
end



function Recurrent:updateOutput(input)
    assert(torch.type(self.state) == 'table' or self.state:dim()>0, "State must be initialized")
    if not self.train then
        self:forget()
    end
    if self.currentIteration == 1 then
        self.initState = recursiveCopy(self.initState, self.state)
    end

    local currentOutput

    if not self.seqMode then
        if self.currentIteration > #self.modules then
            self:setIterations(self.currentIteration)
        end
        self:resizeStateBatch(input:size(1))
        currentOutput = self.modules[self.currentIteration]:forward({input, self.state})
        self.output = currentOutput[1]
        if self.train then
            self.currentIteration = self.currentIteration + 1
        end
    else --sequence mode
        local output = {}
        local __input = input
        if torch.isTensor(input) then --split a time tensor into table
            __input = self.splitInput:forward(input)
        end

        self:resizeStateBatch(__input[1]:size(1))
        currentOutput = {__input[1], self.state}

        if #__input + self.currentIteration - 1 > #self.modules then
            self:setIterations(#__input + self.currentIteration - 1)
        end

        for i=1,#__input do
            local nextInput = {__input[i], currentOutput[2]}
            currentOutput = self.modules[self.currentIteration]:updateOutput(nextInput)
            output[i] =  currentOutput[1]
            self.currentIteration = self.currentIteration + 1
        end

        if torch.isTensor(input) then --join a table into a time tensor
            for i=1,#output do
                output[i] = nn.utils.addSingletonDimension(output[i], self.timeDimension)
            end
            output = self.joinOutput:forward(output)
        end

        self.output = output
    end

    self.state = recursiveCopy(self.state, currentOutput[2])


    self:zeroGradState()

    return self.output, self.state
end

function Recurrent:updateGradInput(input, gradOutput)
    assert(self.train, "must be in training mode")
    self.currentIteration = self.currentIteration - 1

    if not self.seqMode then
        self.gradInput = self.modules[self.currentIteration]:updateGradInput({input, self.state}, {gradOutput, self.gradState})[1]
        self.gradState = self.modules[self.currentIteration].gradInput[2]
    else
        local currentIteration = self.currentIteration --to not disrupt accGradParameters
        local gradInput = {}
        local __input = input
        if torch.isTensor(input) then --split a time tensor into table
            __input = self.splitInput:forward(input)
            gradOutput = gradOutput:split(1, self.timeDimension)
        end

        local currentModule = self.modules[self.currentIteration]
        local currentGradState = self.gradState
        for i=#__input-1,1,-1 do
            currentIteration = currentIteration - 1
            local previousModule = self.modules[currentIteration]
            local previousState = previousModule.output[2]
            currentModule.gradInput = currentModule:updateGradInput({__input[i+1], previousState}, {gradOutput[i+1], currentGradState}, scale)
            gradInput[i+1] = currentModule.gradInput[1]
            currentGradState = currentModule.gradInput[2]
            currentModule = previousModule
        end
        currentModule.gradInput = currentModule:updateGradInput({__input[1], self.initState}, {gradOutput[1], currentGradState}, scale)
        gradInput[1] = currentModule.gradInput[1]
        self.gradState = currentModule.gradInput[2]

        if torch.isTensor(input) then --join a table into a time tensor
            gradInput = self.splitInput:backward(input, gradInput)
        end
        self.gradInput = gradInput
    end
    return self.gradInput, self.gradState
end

function Recurrent:accGradParameters(input, gradOutput, scale)
    assert(self.train, "must be in training mode")
    local scale = scale or 1
    self.currentIteration = self.currentIteration - 1

    if not self.seqMode then
        self.modules[self.currentIteration]:accGradParameters({input, self.state}, {gradOutput, self.gradState}, scale)
        self.gradState = self.modules[self.currentIteration].gradInput[2]
    else
        local __input = input
        if torch.isTensor(input) then --split a time tensor into table
            __input = self.splitInput:forward(input)
            gradOutput = gradOutput:split(1, self.timeDimension)
        end

        local currentGradOutput = {gradOutput[#input], self.gradState}
        local currentModule = self.modules[self.currentIteration]
        local currentGradState = self.gradState
        for i=#__input-1,1,-1 do
            self.currentIteration = self.currentIteration - 1
            local previousModule = self.modules[self.currentIteration]
            local previousState = previousModule.output[2]
            currentModule:accGradParameters({__input[i], previousState}, {gradOutput[i+1], currentGradState}, scale)
            currentGradState = currentModule.gradInput[2]
            currentModule = previousModule
        end
        currentModule:accGradParameters({__input[1], self.initState}, {gradOutput[1], currentGradState}, scale)
        self.gradState = currentModule.gradInput[2]
    end
end

function Recurrent:backward(input, gradOutput, scale)
    assert(self.train, "must be in training mode")
    self.currentIteration = self.currentIteration - 1

    if (torch.type(self.gradState) ~= 'table' and self.gradState:dim() == 0) then
        self:zeroGradState()
    end

    local scale = scale or 1
    if not self.seqMode then
        self.gradInput = self.modules[self.currentIteration]:backward({input, self.state}, {gradOutput, self.gradState}, scale)[1]
        self.gradState = self.modules[self.currentIteration].gradInput[2]
    else
        local gradInput = {}
        local __input = input
        if torch.isTensor(input) then --split a time tensor into table
            __input = self.splitInput:forward(input)
            gradOutput = gradOutput:split(1, self.timeDimension)
        end

        local currentModule = self.modules[self.currentIteration]
        local currentGradState = self.gradState
        for i=#__input-1,1,-1 do
            self.currentIteration = self.currentIteration - 1
            local previousModule = self.modules[self.currentIteration]
            local previousState = previousModule.output[2]
            currentModule.gradInput = currentModule:backward({__input[i+1], previousState}, {gradOutput[i+1], currentGradState}, scale)
            gradInput[i+1] = currentModule.gradInput[1]
            currentGradState = currentModule.gradInput[2]
            currentModule = previousModule
        end
        currentModule.gradInput = currentModule:backward({__input[1], self.initState}, {gradOutput[1], currentGradState}, scale)
        gradInput[1] = currentModule.gradInput[1]
        self.gradState = currentModule.gradInput[2]

        if torch.isTensor(input) then --join a table into a time tensor
            gradInput = self.splitInput:backward(input, gradInput)
        end
        self.gradInput = gradInput
    end
    return self.gradInput, self.gradState
end

function Recurrent:zeroGradParameters()
    self.modules[1]:zeroGradParameters()
end

function Recurrent:updateParameters(learningRate)
    self.modules[1]:updateParameters(learningRate)
end

function Recurrent:clone(...)
    return nn.RecurrentContainer(self.modules[1]:clone(...))
end

function Recurrent:share(m,...)
    return self.modules[1]:share(m.modules[1],...)
end


function Recurrent:reset(stdv)
    self.modules[1]:reset(stdv)
end

function Recurrent:parameters()
    return self.modules[1]:parameters()
end

function Recurrent:shareWeights()
    for i=2,#self.modules do
        self.modules[i]:share(self.modules[1],'weight','bias','gradWeight','gradBias','running_mean','running_std')
    end
end

function Recurrent:training()
    parent.training(self)
    self:forget()
    return self
end

function Recurrent:evaluate()
    parent.evaluate(self)
    self:forget()
    return self
end

function Recurrent:type(t)
    parent.type(self, t)
    self:shareWeights()
    return self
end

function Recurrent:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    return self.name or 'nn.RecurrentContainer {' .. line .. self.modules[1]:__tostring__() .. line .. '}'
end
