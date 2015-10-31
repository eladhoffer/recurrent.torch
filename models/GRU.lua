require 'nngraph'
--[[
adapted from: https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU(inputSize, hiddenSize, n, dropout, initWeights, forgetBias)
    dropout = dropout or 0
    -- there will be 2 input: {input, state}
    -- where state is a table

    local input = nn.Identity()()
    local state = nn.Identity()()

    local initState = {}

    function new_input_sum(insize, xv, hv)
        local i2h = nn.Linear(insize, hiddenSize)(xv)
        local h2h = nn.Linear(hiddenSize, hiddenSize)(hv)
        return nn.CAddTable()({i2h, h2h})
    end

    local x, inputSize_L
    local newState = {}
    local prevState = {state:split(n)}
    for L = 1,n do

        local prev_h = prevState[L]
        -- the input to this layer
        if L == 1 then
            x = input
            inputSize_L = inputSize
        else
            x = newState[L-1]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            inputSize_L = hiddenSize
        end
        -- GRU tick
        -- forward the update and reset gates
        local update_gate = nn.Sigmoid()(new_input_sum(inputSize_L, x, prev_h))
        local reset_gate = nn.Sigmoid()(new_input_sum(inputSize_L, x, prev_h))
        -- compute candidate hidden state
        local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
        local p2 = nn.Linear(hiddenSize, hiddenSize)(gated_hidden)
        local p1 = nn.Linear(inputSize_L, hiddenSize)(x)
        local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        -- compute new interpolated hidden state, based on the update gate
        local zh = nn.CMulTable()({update_gate, hidden_candidate})
        local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
        local next_h = nn.CAddTable()({zh, zhm1})
        table.insert(initState, torch.zeros(hiddenSize))
        table.insert(newState, next_h)
    end
    -- set up the decoder
    local top_h = newState[#newState]
    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

    local rnnModule = nn.gModule({input, state}, {top_h, nn.Identity()(newState)})
    rnnModule:apply( --initialize recurrent weights
    function(m)
        if m.weight then
            m.weight:uniform(-initWeights, initWeights)
        end
        if m.bias then
            m.bias:uniform(-initWeights, initWeights)
        end
    end
    )
    return {
        rnnModule = rnnModule,
        initState = initState
    }
end

return GRU
