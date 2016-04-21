require 'torch'
require 'nn'
require 'optim'
require 'eladtools'
require 'recurrent'
require 'utils.textDataProvider'
-------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training recurrent networks on word-level text dataset - Penn Treebank')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Data Options')
cmd:option('-shuffle',            false,                       'shuffle training samples')

cmd:text('===>Model And Training Regime')
cmd:option('-model',              'LSTM',                      'Recurrent model [RNN, iRNN, LSTM, GRU]')
cmd:option('-seqLength',          50,                          'number of timesteps to unroll for')
cmd:option('-rnnSize',            128,                         'size of rnn hidden layer')
cmd:option('-numLayers',          2,                           'number of layers in the LSTM')
cmd:option('-dropout',            0.5,                           'dropout p value')
cmd:option('-LR',                 2e-3,                        'learning rate')
cmd:option('-LRDecay',            0,                           'learning rate decay (in # samples)')
cmd:option('-weightDecay',        0,                           'L2 penalty on the weights')
cmd:option('-momentum',           0,                           'momentum')
cmd:option('-batchSize',          50,                          'batch size')
cmd:option('-decayRate',          2,                           'exponential decay rate')
cmd:option('-initWeight',         0.08,                        'uniform weight initialization range')
cmd:option('-earlyStop',          5,                           'number of bad epochs to stop after')
cmd:option('-optimization',       'rmsprop',                   'optimization method')
cmd:option('-gradClip',           5,                           'clip gradients at this value')
cmd:option('-epoch',              100,                         'number of epochs to train')
cmd:option('-epochDecay',         5,                           'number of epochs to start decay learning rate')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                           'number of threads')
cmd:option('-type',               'cuda',                      'float or cuda')
cmd:option('-devid',              1,                           'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                           'num of gpu devices used')
cmd:option('-seed',               123,                         'torch manual random number generator seed')
cmd:option('-constBatchSize',     false,                       'do not allow varying batch sizes')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                          'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),      'save directory')
cmd:option('-optState',           false,                       'Save optimization state every epoch')
cmd:option('-checkpoint',         0,                           'Save a weight check point every n samples. 0 for off')




opt = cmd:parse(arg or {})
opt.save = paths.concat('./Results', opt.save)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
local trainWordVec, testWordVec, valWordVec, decoder, decoder_, vocab

trainWordVec, vocab, decoder = loadTextFileWords('./data/ptb.train.txt')
testWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.test.txt', vocab)
assert(#decoder == #decoder_) --no new words
valWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.valid.txt', vocab)
assert(#decoder == #decoder_) --no new words
data = {
  trainingData = trainWordVec,
  testData = testWordVec,
  validationData = valWordVec,
  vocabSize = #decoder,
  decoder = decoder,
  vocab = vocab,
  decode = decodeFunc(vocab, 'word'),
  encode = encodeFunc(vocab, 'word')
}
local vocabSize = #decoder
----------------------------------------------------------------------

if paths.filep(opt.load) then
    modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
else
    modelConfig = {}
    local rnnTypes = {LSTM = nn.LSTM, RNN = nn.RNN, GRU = nn.GRU, iRNN = nn.iRNN}
    local rnn = rnnTypes[opt.model]
    local hiddenSize = opt.rnnSize
    modelConfig.recurrent = nn.Sequential()
    for i=1, opt.numLayers do
      modelConfig.recurrent:add(rnn(hiddenSize, opt.rnnSize, opt.initWeight))
     -- modelConfig.recurrent:add(nn.TemporalModule(nn.BatchNormalization(opt.rnnSize)))
      if opt.dropout > 0 then
        modelConfig.recurrent:add(nn.Dropout(opt.dropout))
      end
      hiddenSize = opt.rnnSize
    end
    modelConfig.embedder = nn.LookupTable(vocabSize, opt.rnnSize)
    modelConfig.classifier = nn.Linear(opt.rnnSize, vocabSize)
end

modelConfig.classifier:share(modelConfig.embedder, 'weight', 'gradWeight')
local trainingConfig = require 'utils.trainRecurrent'
local train = trainingConfig.train
local evaluate = trainingConfig.evaluate
local sample = trainingConfig.sample
local optimState = trainingConfig.optimState
local saveModel = trainingConfig.saveModel

local logFilename = paths.concat(opt.save,'LossRate.log')
local log = optim.Logger(logFilename)
local decreaseLR = EarlyStop(1,opt.epochDecay)
local stopTraining = EarlyStop(opt.earlyStop, opt.epoch)
local epoch = 1

repeat
  print('\nEpoch ' .. epoch ..'\n')
  LossTrain = train(data.trainingData)
  saveModel(epoch)
  if opt.optState then
    torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
  end
  print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

  local LossVal = evaluate(data.validationData)

  print('\nValidation Perplexity: ' .. torch.exp(LossVal))

  local LossTest = evaluate(data.testData)

  --print('\nSampled Text:\n' .. sample('the meaning of life is', 50, true))

  print('\nTest Perplexity: ' .. torch.exp(LossTest))
  log:add{['Training Loss']= LossTrain, ['Validation Loss'] = LossVal, ['Test Loss'] = LossTest}
  log:style{['Training Loss'] = '-', ['Validation Loss'] = '-', ['Test Loss'] = '-'}
  log:plot()
  epoch = epoch + 1

  if decreaseLR:update(LossVal) then
    optimState.learningRate = optimState.learningRate / opt.decayRate
    print("Learning Rate decreased to: " .. optimState.learningRate)
    decreaseLR = EarlyStop(1,1)
    decreaseLR:reset()
  end

until stopTraining:update(LossVal)

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)
