# Recurrent Container for Torch nn modules
This is a simple and efficient way to create recurrent nn modules in Torch. It consists of a container **nn.RecurrentContainer**, which holds a recurrent module.
This recurrent module is expected to receive an input of: `{input, state}` and outputs `{output, state}`. This way, a recurrent module updates its state while giving an output.

For example, a simple RNN module using `nngraph` can be:
```lua
require 'nngraph'
local input = nn.Identity()()
local prevState = nn.Identity()()
local joined = nn.JoinTable(1,1)({input, prevState})
local linear = nn.Linear(inputSize + hiddenSize, hiddenSize)(joined)
local nextState = nn.ReLU()(linear)
local output = nn.Linear(hiddenSize, outputSize)(nextState)

local rnnModule = nn.gModule({input, prevState}, {output, nextState})
```
And the Recurrent container will be configured as:
```lua
local recurrent = nn.RecurrentContainer(rnnModule)
```
## Usage
### Recurrent container modes
There are two ways to use a recurrent container:

**single** - forwarding one time step and getting a subsequent single output

**sequence** - forwarding a sequence of time steps (can be either a table of time steps, or a tensor with a time dimension).

you can change the mode by

`recurrent:single(), recurrent:sequence()`
 or
 `recurrent:mode('single\sequence')`

## Initializing recurrent state
Before usage, the recurrent state must be initialized by using:

`recurrent:setState(initState)`

if used in a batch regime, the initial state can be duplicated using

`recurrent:setState(initState ,batchSize)`

Another way is to use

`recurrent:zeroState([stateSize])`

## Forwarding time series data

The recurrent container is simple to use with other time domain layers.
For example, by using `nn.LookupTable() + nn.TemporalConvolution()` and the included `TemporalCriterion` we can easily configure a model that receives a sentence of length **T** and outputs **T** subsequent classifications to be trained on.
```lua

recurrent:sequence()
recurrent:setState(torch.zeros(hiddenSize), batchSize)

model = nn.Sequential()
model:add(nn.LookupTable(vocabSize, embeddingSize))
model:add(recurrent)
model:add(nn.TemporalConvolution(hiddenSize, numClasses))

criterion = nn.TemporalCriterion(nn.CrossEntropyCriterion())

input = torch.rand(batchSize, T)
output = model:forward(input)
loss = criterion:forward(output, trueLabels)
```
## Remembering time-steps and BPTT

The recurrent container will do backprop-through-time automatically using

`recurrent:backward(input, gradOutput)`

To do so, memory will be allocated for each step forwarded in training mode, and will be removed during a backward step.

To explicitly forget recorded time-steps, use

`recurrent:forget([releaseMemory])`

## License

MIT
