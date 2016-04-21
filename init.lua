require 'torch'
require 'nn'

recurrent = {}
recurrent.rnnModules = {}
require('recurrent.utils')
require('recurrent.Recurrent')
require('recurrent.TemporalModule')
require('recurrent.TemporalCriterion')
require('recurrent.MaskPaddingCriterion')
require('recurrent.Reverse')

require('recurrent.GRU')
require('recurrent.LSTM')
require('recurrent.MultiLayerLSTM')
require('recurrent.RNN')
require('recurrent.iRNN')
require('recurrent.Attention')

--registers all recurrent model under 'recurrent'
for name, func in pairs(recurrent.rnnModules) do
  nn[name] = function(...) return nn.RecurrentContainer(func(...)) end
end

require('recurrent.cudnnUtils') --replaces available modules with cudnn counterpart


-------Adding recurrent function to nn.Container---------------
local Container = nn.Container
function Container:applyRecurrent(func, ...)
    local retVals = {}
    for _,m in pairs(self:listModules()) do    
        if m.__typename == 'nn.RecurrentContainer' then
            local val = m[func](m, ...)
            table.insert(retVals, val)
        end
    end

    return retVals
end


local containerAddfunctions = {
   'setMode', 'sequence', 'single', 'getState',
   'zeroGradState', 'getGradState', 'forget',
   'resizeStateBatch', 'setIterations', 'zeroState', 'isStateful'
  }

for _,fname in pairs(containerAddfunctions) do
  Container[fname] = function(self, ...)
    return self:applyRecurrent(fname, ...)
  end
end

function Container:setState(sourceState)
    local targetState = self:getState()
    recurrent.utils.recursiveCopy(targetState, sourceState)
end

function Container:setGradState(sourceState)
    local targetState = self:getGradState()
    recurrent.utils.recursiveCopy(targetState, sourceState)
end

function Container:accGradState(sourceState)
    local targetState = self:getGradState()
    recurrent.utils.recursiveAdd(targetState, sourceState)
end
--
--function Container:shareGradState(container)
--    local states = container:getGradState()
--    local rnns = self:findModules('nn.RecurrentContainer')
--    assert(#rnns == #states, "Both container should have same amount of recurrent states")
--    for i, rnn in pairs(rnns) do
--        rnn:setGradState(states[i])
--    end
--end
return recurrent
