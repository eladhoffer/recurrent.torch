#Recurrent Container for Torch nn modules
This is a simple and efficient way to create recurrent nn modules in Torch. It consists of a container **RecurrentContainer**, which holds a recurrent module.
This recurrent module is expected to receive an input of: `{input, state}` and outputs `{output, state}`. This way, a recurrent module is expected to receive an input and the
current state, and updates its state while giving an output.
For example, a simple RNN module can be:
```lua
local rnn = nn.Sequential()
rnn:add(nn.JoinTable(1,1))
rnn:add(nn.Linear(inputSize+hiddenSize, hiddenSize))
rnn:add(nn.ReLU())
local outputConcat = nn.ConcatTable()
outputConcat:add(nn.Sequential():add(nn.Linear(hiddenSize, outputSize)):add(nn.LogSoftMax()))
outputConcat:add(nn.Identity()))
rnn:add(outputConcat)
```
And the Recurrent container will be configured as:
```lua
local recurrent = RecurrentContainer(rnnModule)
```

###Recurrent container modes
There are two ways to use a recurrent container:

**single** - forwarding one time step and getting a subsequent single output

**sequence** - forwarding a sequence of time steps (can be either a table of time steps, or a tensor with a time dimension)
## License

MIT
