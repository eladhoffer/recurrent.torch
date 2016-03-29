local Recurrent, parent = torch.class('nn.RecurrentContainer', 'nn.Container')

function Recurrent:__init(recurrentModule)
    parent.__init(self)
    self.modules = {nn.Sequential()}
    self.state = torch.Tensor()
    self.gradState = torch.Tensor()
    self.initState = torch.Tensor()
    self.stateful = true
    self.currentIteration = 1
    self.seqMode = false
    self:setTimeDim(2)
    if recurrentModule then
      self:add(recurrentModule)
    end
end

function Recurrent:setTimeDim(timeDim) --mode can be 'sequence' or 'single'
    self.timeDimension = timeDim
    self.splitInput = nn.SplitTable(self.timeDimension)
    self.joinOutput = nn.JoinTable(self.timeDimension)
    return self
end

function Recurrent:setMode(mode) --mode can be 'sequence' or 'single'
    assert(mode == 'sequence' or mode == 'single')
    self.seqMode = (mode == 'sequence')
    return self
end

function Recurrent:isStateful(mode)
    self.stateful = mode
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
        self.modules[i] = self.modules[1]:clone('weight','bias','gradWeight','gradBias','running_mean','running_std', 'running_var')
    end
end

function Recurrent:add(m, forwardStateModule)
    if torch.type(m) =='table' and (m.rnnModule) and (m.initState) then
        self:setState(m.initState)
        self.name = m.name
        m = m.rnnModule
    elseif forwardStateModule then --will forward the state as is (useful for output projection)
        self.name = (self.name) and self.name .. ' -> ' .. tostring(m)
        m = nn.ParallelTable():add(m):add(nn.Identity())
    end

    self.modules[1]:add(m)
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
        self.state = recurrent.utils.recursiveBatchExpand(self.state, state, batchSize)
    else
        self.state = recurrent.utils.recursiveCopy(self.state, state)
    end
end

function Recurrent:getState()
    return self.state
end

function Recurrent:resizeStateBatch(batchSize)
    self.initState = recurrent.utils.recursiveBatchResize(self.initState, batchSize)
    self.state = recurrent.utils.recursiveBatchResize(self.state, batchSize)
end

function Recurrent:setGradState(gradState)
    self.gradState = recurrent.utils.recursiveCopy(self.gradState, gradState)
end

function Recurrent:accGradState(gradState)
    self.gradState = nn.utils.recursiveAdd(self.gradState, gradState)
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

function Recurrent:__updateOneTimeStep(input)
  if self.currentIteration == 1 then
      self.initState = recurrent.utils.recursiveCopy(self.initState, self.state)
  end
  if self.currentIteration > #self.modules then
      self:setIterations(self.currentIteration)
  end
  self:resizeStateBatch(recurrent.utils.batchSize(input))
  local currentOutput = self.modules[self.currentIteration]:forward({input, self.state})
  self.currentIteration = self.currentIteration + 1
  return currentOutput[1], currentOutput[2]
end

function Recurrent:updateOutput(input)
    assert(torch.type(self.state) == 'table' or self.state:dim()>0, "State must be initialized")
    if not self.train then
        self:forget()
    end
    if not self.stateful then
      self:zeroState()
    end

    local currentOutput

    if not self.seqMode then
        self.output, self.state = self:__updateOneTimeStep(input)
    else --sequence mode
        local output = {}
        local __input = input
        if torch.isTensor(input) then --split a time tensor into table
            __input = self.splitInput:forward(input)
        end

        if #__input + self.currentIteration - 1 > #self.modules then
            self:setIterations(#__input + self.currentIteration - 1)
        end

        for i=1,#__input do
            output[i], self.state = self:__updateOneTimeStep(__input[i])
        end

        if torch.isTensor(input) then --join a table into a time tensor
            for i=1,#output do
                output[i] = nn.utils.addSingletonDimension(output[i], self.timeDimension)
            end
            output = self.joinOutput:forward(output)
        end
        self.output = output
    end

    self:zeroGradState()

    return self.output
end



function Recurrent:__backwardOneStep(input, gradOutput, scale)
  self.currentIteration = self.currentIteration - 1
  local currentModule = self.modules[self.currentIteration]
  local previousState = self.initState

  if self.currentIteration > 1 then
    previousState = self.modules[self.currentIteration - 1].output[2]
  end
  currentModule:backward({input, previousState}, {gradOutput, self.gradState}, scale)

  return currentModule.gradInput[1], currentModule.gradInput[2]
end

function Recurrent:backward(input, gradOutput, scale)
    assert(self.train, "must be in training mode")
    if (torch.type(self.gradState) ~= 'table' and self.gradState:dim() == 0) then
        self:zeroGradState()
    end

    local scale = scale or 1
    if not self.seqMode then
        self.gradInput, self.gradState = self:__backwardOneStep(input, gradOutput, scale)
    else
        local gradInput = {}
        local __input = input
        if torch.isTensor(input) then --split a time tensor into table
            __input = self.splitInput:forward(input)
            gradOutput = gradOutput:split(1, self.timeDimension)
        end

        for i=#__input,1,-1 do
            gradInput[i], self.gradState = self:__backwardOneStep(__input[i], gradOutput[i], scale)
        end

        if torch.isTensor(input) then --join a table into a time tensor
            gradInput = self.splitInput:backward(input, gradInput)
        end
        self.gradInput = gradInput
    end
    return self.gradInput
end

function Recurrent:zeroGradParameters()
    self.modules[1]:zeroGradParameters()
end

function Recurrent:updateParameters(learningRate)
    self.modules[1]:updateParameters(learningRate)
end

function Recurrent:clone(...)
    local m = nn.RecurrentContainer(self.modules[1]:clone(...))
    m:setState(self:getState())
    m:setGradState(self:getGradState())
    m.name = self.name
    return m
end

function Recurrent:share(m,...)
    return self.modules[1]:share(m.modules[1],...)
end

function Recurrent:reset(stdv)
    self:zeroState()
    self:zeroGradState()
    self.modules[1]:reset(stdv)
end

function Recurrent:parameters()
    return self.modules[1]:parameters()
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

function Recurrent:type(t, tensorCache)
    local tensorCache = tensorCache or {}
    parent.type(self, t, tensorCache)
    return self
end

function Recurrent:clearState()
   self:setIterations(1, true)
   self:resizeStateBatch(1)
   self.splitInput:clearState()
   self.joinOutput:clearState()
   nn.utils.clear(self, {'gradState', 'initState'})
   return parent.clearState(self)
end

function Recurrent:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    return self.name or 'nn.RecurrentContainer {' .. line .. self.modules[1]:__tostring__() .. line .. '}'
end
