require 'nngraph'
--[[
adapted from: https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
local function GRU(inputSize, outputSize, initWeights, forgetBias)
    -- there will be 2 input: {input, state}
    local initWeights = initWeights or 0.08
    local input = nn.Identity()()
    local state = nn.Identity()()


    function new_input_sum(insize, xv, hv)
        local i2h = nn.Linear(insize, outputSize)(xv)
        local h2h = nn.Linear(outputSize, outputSize)(hv)
        return nn.CAddTable()({i2h, h2h})
    end

    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(inputSize, input, state))
    local reset_gate = nn.Sigmoid()(new_input_sum(inputSize, input, state))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, state})
    local p2 = nn.Linear(outputSize, outputSize)(gated_hidden)
    local p1 = nn.Linear(inputSize, outputSize)(input)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), state})
    local nextState = nn.CAddTable()({zh, zhm1})

    local rnnModule = nn.gModule({input, state}, {nextState, nn.Identity()(nextState)})
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
        initState = torch.zeros(1, outputSize),
        name = 'GRU: ' .. inputSize .. ' -> ' .. outputSize .. ', ' .. outputSize
    }
end

recurrent.rnnModules['GRU'] = GRU
return GRU
