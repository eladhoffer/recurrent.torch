require 'torch'
require 'nn'

recurrent = {}
torch.include('recurrent', 'Recurrent.lua')
torch.include('recurrent', 'TemporalCriterion.lua')
torch.include('recurrent', 'GRU.lua')
torch.include('recurrent', 'LSTM.lua')
torch.include('recurrent', 'RNN.lua')
return recurrent
