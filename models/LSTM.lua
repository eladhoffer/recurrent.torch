require 'nngraph'
--adapted from: https://github.com/karpathy/char-rnn
local function LSTM(inputSize, outputSize, initWeights, forgetBias)
    local forgetBias = forgetBias or 1
    local initWeights = initWeights or 0.08
    -- there will be 2 input: {input, state}
    -- where state is concatenated
    local input = nn.Identity()()
    local state = nn.Identity()()

    -- c,h from previos timesteps
    local prev_c, prev_h = nn.SplitTable(1,2)(nn.Reshape(2, outputSize)(state)):split(2)

    -- evaluate the input sums at once for efficiency
    local input_prev_h = nn.JoinTable(1,1)({input, prev_h})
    local all_input_sums = nn.Linear(inputSize + outputSize, 4 * outputSize)(input_prev_h)

    local reshaped = nn.Reshape(4, outputSize)(all_input_sums)
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

    local nextState = nn.JoinTable(2)({next_c, next_h})
    local model = nn.gModule({input, state}, {next_h, nextState})

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
      all_input_sums.data.module.bias:narrow(1, outputSize+1, outputSize):fill(forgetBias)
    end

    return {
        rnnModule = model,
        initState = torch.zeros(1, 2 * outputSize),
        name = 'nn.LSTM(' .. inputSize .. ' -> ' .. outputSize .. ', ' .. 2 * outputSize .. ')'
    }
end

recurrent.rnnModules['LSTM'] = LSTM
return LSTM
