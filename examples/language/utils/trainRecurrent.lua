require 'optim'
require 'eladtools'
require 'nn'
require 'recurrent'
----------------------------------------------------------------------
-- Output files configuration
os.execute('mkdir -p ' .. opt.save)

cmd:log(opt.save .. '/log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local optStateFilename = paths.concat(opt.save,'optState')

local vocabSize = data.vocabSize
local vocab = data.vocab
local decoder = data.decoder



local trainRegime = modelConfig.regime
local recurrent = modelConfig.recurrent
local stateSize = modelConfig.stateSize
local embedder = modelConfig.embedder
local classifier = modelConfig.classifier

-- Model + Loss:
local model = nn.Sequential()
model:add(embedder)
model:add(recurrent)
model:add(nn.TemporalModule(classifier))
local criterion = nn.CrossEntropyCriterion()--ClassNLLCriterion()

local TensorType = 'torch.FloatTensor'


if opt.type =='cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)

    model:cuda()

    criterion = criterion:cuda()
    TensorType = 'torch.CudaTensor'


    ---Support for multiple GPUs - currently data parallel scheme
    if opt.nGPU > 1 then
        initState:resize(opt.batchSize / opt.nGPU, stateSize)
        local net = model
        model = nn.DataParallelTable(1)
        model:add(net, 1)
        for i = 2, opt.nGPU do
            cutorch.setDevice(i)
            model:add(net:clone(), i)  -- Use the ith GPU
        end
        cutorch.setDevice(opt.devid)
    end
end

--sequential criterion
local seqCriterion = nn.TemporalCriterion(criterion)

-- Optimization configuration
local Weights,Gradients = model:getParameters()

--initialize weights uniformly
Weights:uniform(-opt.initWeight, opt.initWeight)

local savedModel = {
    embedder = embedder:clone('weight','bias', 'running_mean', 'running_std'),
    recurrent = recurrent:clone('weight','bias', 'running_mean', 'running_std'),
    classifier = classifier:clone('weight','bias', 'running_mean', 'running_std')
}

----------------------------------------------------------------------
print '\n==> Network'
print(model)
print('\n==>' .. Weights:nElement() ..  ' Parameters')

print '\n==> Criterion'
print(criterion)

if trainRegime then
    print '\n==> Training Regime'
    table.foreach(trainRegime, function(x, val) print(string.format('%012s',x), unpack(val)) end)
end
------------------Optimization Configuration--------------------------

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

local optimizer = Optimizer{
    Model = model,
    Loss = seqCriterion,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
    Regime = trainRegime,
    GradRenorm = opt.gradClip
}

----------------------------------------------------------------------
---utility functions


local function reshapeData(wordVec, seqLength, batchSize)
    local offset = offset or 0
    local length = wordVec:nElement()
    local numBatches = torch.floor(length / (batchSize * seqLength))

    local batchWordVec = wordVec.new():resize(numBatches, batchSize, seqLength)
    local endWords = wordVec.new():resize(numBatches, batchSize, 1)

    local endIdxs = torch.LongTensor()
    for i=1, batchSize do
        local startPos = torch.round((i - 1) * length / batchSize ) + 1
        local sliceLength = seqLength * numBatches
        local endPos = startPos + sliceLength - 1

        batchWordVec:select(2,i):copy(wordVec:narrow(1, startPos, sliceLength))
    endIdxs:range(startPos + seqLength, endPos + 1, seqLength)
endWords:select(2,i):copy(wordVec:index(1, endIdxs))
  end
  return batchWordVec, endWords
end

local function saveModel(epoch)
    local fn = netFilename .. '_' .. epoch .. '.t7'
    torch.save(fn,
    {
        embedder = savedModel.embedder:clone():float(),
        recurrent = savedModel.recurrent:clone():float(),
        classifier = savedModel.classifier:clone():float(),
        inputSize = inputSize,
        stateSize = stateSize,
        vocab = vocab,
        decoder = decoder
    })
    collectgarbage()
end

----------------------------------------------------------------------
local function ForwardSeq(dataVec, train)

    local data, labels = reshapeData(dataVec, opt.seqLength, opt.batchSize )
    local sizeData = data:size(1)
    local numSamples = 0
    local lossVal = 0
    local currLoss = 0
    local x = torch.Tensor(opt.batchSize, opt.seqLength):type(TensorType)
    local yt = torch.Tensor(opt.batchSize, opt.seqLength):type(TensorType)

    -- input is a sequence
    model:sequence()
    model:forget()

    for b=1, sizeData do
        if b==1 or opt.shuffle then --no dependancy between consecutive batches
            model:zeroState()
        end
        x:copy(data[b])
        yt:narrow(2,1,opt.seqLength-1):copy(x:narrow(2,2,opt.seqLength-1))
        yt:select(2, opt.seqLength):copy(labels[b])

        if train then
            if opt.nGPU > 1 then
                model:syncParameters()
            end
            y, currLoss = optimizer:optimize(x, yt)
        else
            y = model:forward(x)
            currLoss = seqCriterion:forward(y,yt)
        end
        lossVal = currLoss / opt.seqLength + lossVal
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, sizeData*opt.batchSize)
    end

    collectgarbage()
    xlua.progress(numSamples, sizeData)
    return lossVal / sizeData
end

local function ForwardSingle(dataVec)
    local sizeData = dataVec:nElement()
    local numSamples = 0
    local lossVal = 0
    local currLoss = 0

    -- input is from a single time step
    model:single()

    local x = torch.Tensor(1,1):type(TensorType)
    local y
    for i=1, sizeData-1 do
        x:fill(dataVec[i])
        y = recurrent:forward(embedder:forward(x):select(2,1))
        currLoss = criterion:forward(y, dataVec[i+1])
        lossVal = currLoss + lossVal
        if (i % 100 == 0) then
            xlua.progress(i, sizeData)
        end
    end

    collectgarbage()
    return(lossVal/sizeData)
end

------------------------------

local function train(dataVec)
    model:training()
    return ForwardSeq(dataVec, true)
end

local function evaluate(dataVec)
    model:evaluate()
    return ForwardSeq(dataVec, false)
end

local function sample(str, num, space, temperature)
    local num = num or 50
    local temperature = temperature or 1
    local function smp(preds)
        if temperature == 0 then
            local _, num = preds:max(2)
            return num
        else
            preds:div(temperature) -- scale by temperature
            local probs = preds:squeeze()
            probs:div(probs:sum()) -- renormalize so probs sum to one
            local num = torch.multinomial(probs:float(), 1):typeAs(preds)
            return num
        end
    end


    recurrent:evaluate()
    recurrent:single()

    local sampleModel = nn.Sequential():add(embedder):add(recurrent):add(classifier):add(nn.SoftMax():type(TensorType))

    local pred, predText, embedded
    if str then
        local encoded = data.encode(str)
        for i=1, encoded:nElement() do
            pred = sampleModel:forward(encoded:narrow(1,i,1))
        end
        wordNum = smp(pred)

        predText = str .. '... ' .. decoder[wordNum:squeeze()]
    else
        wordNum = torch.Tensor(1):random(vocabSize):type(TensorType)
        predText = ''
    end

    for i=1, num do
        pred = sampleModel:forward(wordNum)
        wordNum = smp(pred)
        if space then
            predText = predText .. ' ' .. decoder[wordNum:squeeze()]
        else
            predText = predText .. decoder[wordNum:squeeze()]
        end
    end
    return predText
end
return {
    train = train,
    evaluate = evaluate,
    sample = sample,
    saveModel = saveModel,
    optimState = optimState,
    model = model
}
