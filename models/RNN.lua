require 'nn'
local function RNN(inputSize, hiddenSize)
    local outputSize = outputSize or inputSize
    local linear = nn.Linear(inputSize+hiddenSize, hiddenSize)
    local rnn = nn.Sequential()
    rnn:add(nn.JoinTable(1,1))
    rnn:add(linear)
    rnn:add(nn.Sigmoid())
    rnn:add(nn.ConcatTable():add(nn.Identity()):add(nn.Identity()))
    return {
      rnnModule = rnn,
      initState = torch.zeros(1, hiddenSize),
      name = 'nn.RNN(' .. inputSize .. ' -> ' .. outputSize .. ', ' .. outputSize .. ')'
    }

end

recurrent.rnnModules['RNN'] = RNN
return RNN
