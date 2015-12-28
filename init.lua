require 'torch'
require 'nn'

recurrent = {}
recurrent.rnnModules = {}
torch.include('recurrent', 'Recurrent.lua')
torch.include('recurrent', 'TemporalModule.lua')
torch.include('recurrent', 'TemporalCriterion.lua')

torch.include('recurrent', 'GRU.lua')
torch.include('recurrent', 'LSTM.lua')
torch.include('recurrent', 'RNN.lua')

--registers all recurrent model under 'recurrent'
for name, func in pairs(recurrent.rnnModules) do
  nn[name] = function(...) return nn.RecurrentContainer(func(...)) end
end



-------Adding recurrent function to nn.Container---------------
local Container = nn.Container
function Container:applyRecurrent(func, ...)
    local retVals = {}
    for i, rnn in pairs(self:findModules('nn.RecurrentContainer')) do
        retVals[i] = rnn[func](rnn, ...)
    end
    return retVals
end


local containerAddfunctions = {
   'setMode', 'sequence', 'single', 'setState', 'getState',
   'setGradState', 'zeroGradState', 'getGradState', 'forget',
   'resizeStateBatch', 'setIterations', 'zeroState'
  }

for _,fname in pairs(containerAddfunctions) do
  Container[fname] = function(self, ...)
    return self:applyRecurrent(fname)
  end
end

function Container:shareState(container)
    local states = container:getState()
    local rnns = self:findModules('nn.RecurrentContainer')
    assert(#rnns == #states, "Both container should have same amount of recurrent states")
    for i, rnn in pairs(rnns) do
        rnn:setState(states[i])
    end
end

return recurrent
