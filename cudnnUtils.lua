if not cudnn or cudnn.version < 5000 then
    print('no cudnn')
    return
end

recurrent.cudnnEnabled = true

local rec = nn.RecurrentContainer

local addFunctions = {
   'setMode', 'sequence', 'single', 'getState',
   'zeroGradState', 'getGradState', 'forget',
   'zeroState', 'isStateful'
  }

function wrapCudnnModule(m)
   m.batchFirst = true
   m.state = {m.hiddenOutput, m.cellOutput}
   m.gradState = {m.gradHiddenInput, m.gradCellInput}
   m.stateful = true
   m.seqMode = true
   m.__typename = 'nn.RecurrentContainer'
    
   for _,f in pairs(addFunctions) do
       m[f] = rec[f]
   end
   m.setIterations = function(self, iterations, clear) return end

   return m
end

local moduleList = {
    RNN = 'RNN',
    LSTM = 'LSTM',
    GRU = 'GRU'
}

for name, cudnnName in pairs(moduleList) do
    nn[name] = function(inputSize, hiddenSize, initWeight, numLayers)
        local numLayers = numLayers or 1
        local m =  wrapCudnnModule(cudnn[cudnnName](inputSize, hiddenSize, numLayers, true))
        if initWeight then
            m.weight:uniform(-initWeight, initWeight)
        end
        return m

    end
end
