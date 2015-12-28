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

return recurrent
