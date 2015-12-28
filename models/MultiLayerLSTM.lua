require 'nngraph'
--adapted from: https://github.com/karpathy/char-rnn
local function MultiLayerLSTM(inputSize, hiddenSize, n, dropout, initWeights, forgetBias)
  local dropout = dropout or 0
  local n = n or 1
  local initWeights = initWeights or 0.08

  -- there will be 2 input: {input, state}
  -- where state is concatenated
  local input = nn.Identity()()

  local state = nn.SelectTable(2)(input)
  local forgetGatesBiases = {}
  local newState = {}

  local x, inputSize_L
  local output = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_c = nn.Narrow(2, 2*(L-1)*hiddenSize + 1, hiddenSize)(state)
    local prev_h = nn.Narrow(2, 2*(L-1)*hiddenSize + hiddenSize + 1, hiddenSize)(state)
    -- the input to this layer
    if L == 1 then
      x = nn.SelectTable(1)(input)
      inputSize_L = inputSize
    else
      x = newState[2*(L-1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      inputSize_L = hiddenSize
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(inputSize_L, 4 * hiddenSize)(x)
    local h2h = nn.Linear(hiddenSize, 4 * hiddenSize)(prev_h)

    --remember forget gate biases for easy initialization
    table.insert(forgetGatesBiases, i2h.data.module.bias:narrow(1, hiddenSize+1, hiddenSize))
    table.insert(forgetGatesBiases, h2h.data.module.bias:narrow(1, hiddenSize+1, hiddenSize))

    local all_input_sums = nn.CAddTable()({i2h, h2h})
    local reshaped = nn.Reshape(4, hiddenSize)(all_input_sums)

    local n1, n2, n3, n4 = nn.SplitTable(1,2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)

    local out_gate = nn.Sigmoid()(n3)
    -- decode the write input
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    newState[2*L-1] = next_c
    newState[2*L] = next_h
  end

  -- set up the decoder
  local top_h = newState[#newState]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local model = nn.gModule({input}, {top_h, nn.JoinTable(2)(newState)})

  model:apply( --initialize recurrent weights
  function(m)
    if m.weight then
      m.weight:uniform(-initWeights, initWeights)
    end
    if m.bias then
      m.bias:uniform(-initWeights, initWeights)
    end
  end
  )
  if forgetBias then
    for _,fGateBias in pairs(forgetGatesBiases) do
      fGateBias:fill(forgetBias)
    end
  end

  return {
    rnnModule = model,
    initState = torch.zeros(1, 2 * n * hiddenSize),
    name = 'MultiLayerLSTM: ' .. inputSize .. ' -> ' .. outputSize .. ', ' .. 2 * n * outputSize
  }
end

recurrent.rnnModules['MultiLayerLSTM'] = MultiLayerLSTM
return MultiLayerLSTM
