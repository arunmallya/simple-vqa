require 'nn'

local SigmoidCrossEntropyCriterion, parent =
torch.class('nn.SigmoidCrossEntropyCriterion', 'nn.Criterion')


function SigmoidCrossEntropyCriterion:__init(weights)
    parent.__init(self)

    self.sigmoid = nn.Sigmoid()
end


function SigmoidCrossEntropyCriterion:updateOutput(input, target)
    local input_size = input:size()
    local x = input:clone()
    local t = target:clone()
    x = x:view(x:nElement())
    t = t:view(t:nElement())

    self.sigmoid:updateOutput(x)
    
    local loss, indicator = 0, 0
    for i = 1, x:size(1) do
      if x[i] >= 0 then
        indicator = 1
      else
        indicator = 0
      end
      loss = loss - x[i] * (t[i] - indicator) + math.log(1 + math.exp(x[i] - 2 * x[i] * indicator))
    end

    self.output = loss / input_size[1]
    return self.output
end


function SigmoidCrossEntropyCriterion:updateGradInput(input, target)
    local t = target:clone()
    t = t:view(t:nElement())
    local output = self.sigmoid.output - t
    output:div(input:size(1))
    output = torch.reshape(output, input:size())
    self.gradInput = output
    return self.gradInput
end