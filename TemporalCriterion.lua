local TemporalCriterion, parent = torch.class('nn.TemporalCriterion', 'nn.Criterion')


function TemporalCriterion:__init(criterion, repeatTarget, evalAsBatch)
  parent.__init(self)
  self.criterion = criterion
  self.gradInput = {}
  self.repeatTarget = repeatTarget
  self.evalAsBatch = evalAsBatch
  if (self.evalAsBatch == nil) then
    self.evalAsBatch = true
  end
end

local function viewTimeAsBatch(input)
  local szInput = input:size():totable()
  szInput[2] = szInput[1]*szInput[2]
  table.remove(szInput, 1)
  return input:view(unpack(szInput))
end

function TemporalCriterion:updateOutput(input, target)
  if torch.isTensor(input) and self.evalAsBatch then
    self.output = self.criterion:updateOutput(viewTimeAsBatch(input), viewTimeAsBatch(target)) * input:size(2)
  else
    self.output = 0
    if torch.type(input) == 'table' then
      for t = 1, #input do
        local target = self.repeatTarget and target or target[i]
        self.output = self.output + self.criterion:updateOutput(input[i], target:squeeze())
      end
    else
      for t = 1, input:size(2) do
        local target = self.repeatTarget and target or target:select(2, t)
        self.output = self.output + self.criterion:updateOutput(input:select(2, t), target)
      end
    end
  end
  return self.output
end

function TemporalCriterion:updateGradInput(input, target)
  if torch.isTensor(input) and self.evalAsBatch then
    self.gradInput = self.criterion:updateGradInput(viewTimeAsBatch(input), viewTimeAsBatch(target)):viewAs(input) * input:size(2)
  else
    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
    nn.utils.recursiveFill(self.gradInput, 0)
    if torch.type(input) == 'table' then
      for t = 1, #input do
        local target = self.repeatTarget and target or target[i]
        nn.utils.recursiveAdd(self.gradInput[i], self.criterion:updateGradInput(input[i], target:squeeze()))
      end
    else
      for t = 1, input:size(2) do
        local target = self.repeatTarget and target or target:select(2, t)
        nn.utils.recursiveAdd(self.gradInput:select(2, t), self.criterion:updateGradInput(input:select(2, t), target))
      end
    end
  end
  return self.gradInput
end

function TemporalCriterion:type(type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end
