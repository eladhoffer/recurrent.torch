local TemporalCriterion, parent = torch.class('nn.TemporalCriterion', 'nn.Criterion')


function TemporalCriterion:__init(criterion, evalAsBatch, repeatTarget)
    parent.__init(self)
    self.criterion = criterion
    self.gradInput = {}
    self.repeatTarget = repeatTarget
    self.evalAsBatch = evalAsBatch
end

local function viewTimeAsBatch(input)
    local szInput = input:size():totable()
    szInput[2] = szInput[1]*szInput[2]
    table.remove(szInput, 1)
    return input:view(unpack(szInput))
end
function TemporalCriterion:updateOutput(input, target)
    if self.evalAsBatch then
        self.output = self.criterion:updateOutput(viewTimeAsBatch(input), viewTimeAsBatch(target)) * input:size(2)
    else
        self.output = 0
        for t = 1, input:size(2) do
            local target = self.repeatTarget and target or target:select(2, t)
            self.output = self.output + self.criterion:updateOutput(input:select(2, t), target)
        end
    end
    return self.output
end

function TemporalCriterion:updateGradInput(input, target)
    if self.evalAsBatch then
        self.gradInput = self.criterion:updateGradInput(viewTimeAsBatch(input), viewTimeAsBatch(target)):viewAs(input) * input:size(2)
    else
        self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
        nn.utils.recursiveFill(self.gradInput, 0)
        for t = 1, input:size(2) do
            local target = self.repeatTarget and target or target:select(2, t)
            nn.utils.recursiveAdd(self.gradInput:select(2, t), self.criterion:updateGradInput(input:select(2, t), target))
        end
    end
    return self.gradInput
end

function TemporalCriterion:type(type, tensorCache)
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end
