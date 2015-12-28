local TemporalModule, parent = torch.class('nn.TemporalModule', 'nn.Sequential')


function TemporalModule:__init(module)
    parent.__init(self)
    self.modules = {module}
end

local function viewTimeAsBatch(input)
    local szInput = input:size():totable()
    szInput[2] = szInput[1]*szInput[2]
    table.remove(szInput, 1)
    return input:view(unpack(szInput))
end

local function viewTime(input, timeLength)
    local szInput = input:size():totable()
    szInput[1] = timeLength
    return input:view(-1, unpack(szInput))
end
function TemporalModule:updateOutput(input)
    local timeLength = input:size(2)
    self.output = viewTime(parent.updateOutput(self, viewTimeAsBatch(input)), timeLength)
    return self.output
end

function TemporalModule:updateGradInput(input, gradOutput,...)
    self.gradInput = parent.updateGradInput(self, viewTimeAsBatch(input), viewTimeAsBatch(gradOutput), ...):viewAs(input)
    return self.gradInput
end

function TemporalModule:accGradParameters(input, gradOutput, ...)
    parent.accGradParameters(self, viewTimeAsBatch(input), viewTimeAsBatch(gradOutput), ...)
end

function TemporalModule:backward(input, gradOutput,...)
    self.gradInput = parent.backward(self, viewTimeAsBatch(input), viewTimeAsBatch(gradOutput), ...):viewAs(input)
    return self.gradInput
end

function TemporalModule:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.TemporalModule'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
