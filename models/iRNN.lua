require 'nn'
local function iRNN(inputSize, hiddenSize)
    local outputSize = outputSize or inputSize
    local linear = nn.Linear(inputSize+hiddenSize, hiddenSize)
    linear.weight:narrow(2,inputSize+1, hiddenSize):copy(torch.eye(hiddenSize))
    linear.bias:zero()
    local rnn = nn.Sequential()
    rnn:add(nn.JoinTable(1,1))
    rnn:add(linear)
    rnn:add(nn.ReLU())
    rnn:add(nn.ConcatTable():add(nn.Identity()):add(nn.Identity()))
    return {
      rnnModule = rnn,
      initState = torch.zeros(1, hiddenSize),
      name = 'iRNN ' .. inputSize .. ' -> ' .. outputSize .. ', ' .. outputSize
    }

end

recurrent.rnnModules['iRNN'] = iRNN

return iRNN
