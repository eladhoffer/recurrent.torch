require 'nn'
local function RNN(inputSize, outputSize)
    local linear = nn.Linear(inputSize + outputSize, outputSize)
    local rnn = nn.Sequential()
    rnn:add(nn.JoinTable(1,1))
    rnn:add(linear)
    rnn:add(nn.Tanh())
    rnn:add(nn.ConcatTable():add(nn.Identity()):add(nn.Identity()))
    return {
      rnnModule = rnn,
      initState = torch.zeros(1, outputSize),
      name = 'nn.RNN(' .. inputSize .. ' -> ' .. outputSize .. ', ' .. outputSize .. ')'
    }

end

recurrent.rnnModules['RNN'] = RNN
return RNN
