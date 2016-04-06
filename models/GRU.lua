require 'nngraph'
--[[
adapted from: https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
local function GRU(inputSize, outputSize, initWeights)
    -- there will be 2 input: {input, state}
    local initWeights = initWeights or 0.08
    local input = nn.Identity()()
    local state = nn.Identity()()

    -- GRU tick
    -- forward the update and reset gates
    local input_state = nn.JoinTable(1,1)({input, state})
    local gates = nn.Sigmoid()(nn.Linear(inputSize + outputSize, 2*outputSize)(input_state))
    local update_gate, reset_gate = nn.SplitTable(1,2)(nn.Reshape(2, outputSize)(gates)):split(2)

    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, state})
    local input_gHidden = nn.JoinTable(1,1)({input, gated_hidden})
    local hidden_candidate = nn.Tanh()(nn.Linear(inputSize + outputSize, outputSize)(input_gHidden))
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
        name = 'nn.GRU(' .. inputSize .. ' -> ' .. outputSize .. ', ' .. outputSize .. ')'
    }
end

recurrent.rnnModules['GRU'] = GRU
return GRU
