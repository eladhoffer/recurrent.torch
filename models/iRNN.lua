require 'nn'
local function iRNN(inputSize, outputSize)
    local linear = nn.Linear(inputSize+outputSize, outputSize)
    linear.weight:narrow(2,inputSize+1, outputSize):copy(torch.eye(outputSize))
    linear.bias:zero()
    local rnn = nn.Sequential()
    rnn:add(nn.JoinTable(2))
    rnn:add(linear)
    rnn:add(nn.ReLU())
    rnn:add(nn.ConcatTable():add(nn.Identity()):add(nn.Identity()))
    return {
      rnnModule = rnn,
      initState = torch.zeros(1, outputSize),
      name = 'nn.iRNN(' .. inputSize .. ' -> ' .. outputSize .. ', ' .. outputSize .. ')'
    }

end

recurrent.rnnModules['iRNN'] = iRNN

return iRNN
