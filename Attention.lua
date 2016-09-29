require 'nn'
require 'nngraph'


local function efficient1x1Conv(inputSize, outputDim, T)
  local module = nn.Sequential()
  module:add(nn.View(-1, inputSize))
  module:add(nn.Linear(inputSize, outputDim))
  if outputDim == 1 then
    module:add(nn.View(-1, T))
  else
    module:add(nn.View(-1, T, outputDim))
  end
  return module
end

local function attention(T, eInputSize, dInputSize)
  local h = nn.Identity()() --h is 3 dimensional: batch x T x eInputSize
  local dt = nn.Identity()() --dt is 2 dimensional: batch x dInputSize
  local W2dt = nn.Linear(dInputSize, eInputSize)(dt)
  local W2dtRepeated = nn.Replicate(T, 2, 2)(W2dt)
  local W1h = efficient1x1Conv(eInputSize, eInputSize, T)(h)--nn.TemporalConvolution(eInputSize, eInputSize, 1, 1)(h)
  local TanSum = nn.Tanh()(nn.CAddTable()({W1h, W2dtRepeated})) -- size: batch x T x eInputSize
  local u = efficient1x1Conv(eInputSize, 1, T)(TanSum) -- size: batch x T
  local a = nn.SoftMax()(u) -- size: batch x T
  local weightedH = nn.CMulTable(){h, nn.Replicate(eInputSize, 2, 1)(a)}
  local dtN = nn.Sum(1, 2)(weightedH) --output size: batch x eInputSize
  return nn.gModule({h,dt}, {dtN, a})
end


local function AttentiveRecurrent(rnnModule, attentionModule)
  local input = nn.Identity()()
  local state = nn.Identity()()
  local attnInput = nn.SelectTable(2)(input)
  local rnnState = nn.SelectTable(1)(state)
  local attnState = nn.SelectTable(2)(state)
  local rnnInput = nn.JoinTable(1,1)({nn.SelectTable(1)(input), attnState})
  local rnnOutput, rnnNewState = rnnModule({rnnInput, rnnState}):split(2)
  local attn = attentionModule({attnInput, rnnState})
  local attnNewState = nn.SelectTable(1)(attn)
  local attnWeights = nn.SelectTable(2)(attn)
  local newState = nn.Identity()({rnnNewState, attnNewState})

  return nn.gModule({input,state}, {rnnOutput, newState, attnWeights})
end

local function AttentiveGRU(inputSize, outputSize, attnSize, attnTime)
  local config = recurrent.rnnModules.GRU(inputSize + attnSize, outputSize)
  local aGRUModule = AttentiveRecurrent(config.rnnModule, attention(attnTime, attnSize, outputSize))
  local name = 'nn.AttentiveGRU({' .. inputSize .. ', ' .. attnTime .. ' x ' .. attnSize .. '} -> '
  name = name  .. outputSize .. ', {' .. outputSize .. ', ' .. attnSize ..'})'
  return {
      rnnModule = aGRUModule,
      initState = {config.initState, torch.zeros(1, attnSize)},
      name = name
  }
end

local function AttentiveLSTM(inputSize, outputSize, attnSize, attnTime)
  local config = recurrent.rnnModules.LSTM(inputSize + attnSize, outputSize)
  local aLSTM = AttentiveRecurrent(config.rnnModule, attention(attnTime, attnSize, 2*outputSize))
  local name = 'nn.AttentiveLSTM({' .. inputSize .. ', ' .. attnTime .. ' x ' .. attnSize .. '} -> '
  name = name  .. outputSize .. ', {' .. 2*outputSize .. ', ' .. attnSize ..'})'
  return {
      rnnModule = aLSTM,
      initState = {config.initState, torch.zeros(1, attnSize)},
      name = name
  }
end


recurrent.rnnModules['AttentiveLSTM'] = AttentiveLSTM
recurrent.rnnModules['AttentiveGRU'] = AttentiveGRU
