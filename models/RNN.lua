require 'nn'
function RNN(inputSize, hiddenSize, n, dropout)
    local outputSize = outputSize or inputSize
    local linear = nn.Linear(inputSize+hiddenSize, hiddenSize)
    local rnn = nn.Sequential()
    rnn:add(nn.JoinTable(1,1))
    rnn:add(linear)
    rnn:add(nn.Sigmoid())
    if dropout > 0 then
      rnn:add(nn.Dropout(dropout))
    end
    rnn:add(nn.ConcatTable():add(nn.Identity()):add(nn.Identity()))
    return {
      rnnModule = rnn,
      initState = torch.zeros(hiddenSize)
    }

end
return RNN
