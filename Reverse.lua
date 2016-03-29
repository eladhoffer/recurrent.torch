require 'nn'
local Reverse, parent = torch.class('nn.Reverse', 'nn.Module')

function Reverse:__init(dim)
  parent.__init(self)
  self.dim = dim or 2
  self.reverseIdx = torch.LongTensor()
end

function Reverse:updateOutput(input)
  self.reverseIdx = self.reverseIdx:long()
  self.reverseIdx:resize(input:size(self.dim)):range(input:size(self.dim), 1, -1)
  self.output:resizeAs(input)
  self.output:indexCopy(self.dim, self.reverseIdx, input)
  return self.output
end

function Reverse:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  self.gradInput:indexCopy(self.dim, self.reverseIdx, gradOutput)
  return self.gradInput
end
