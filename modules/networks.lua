require 'nn'
require 'nngraph'
local nninit = require 'nninit'


local networks = {}


function networks:getNetwork(net_type, net_inputs, im_feat_dims)
  if net_type == 'Linear' then
    return networks:Linear()
  elseif net_type == 'Bilinear' then
    return networks:Bilinear()
  elseif net_type == 'MLP' then 
    if net_inputs == 'iqa' then
      assert(#im_feat_dims == 1)
      return networks:MLP({im_feat_dims[1], 300, 300})
    elseif net_inputs == 'qa' then
      return networks:MLP_qa()
    end
  else 
    error(net_type .. ' not implemented!')
  end
end


function networks:Linear()
  local im_feat = nn.Identity()()
  local q_feat = nn.Identity()()
  local choice_feat = nn.Identity()()
  local all_feat = nn.JoinTable(2)({im_feat, q_feat, choice_feat})
  local fc1 = nn.Linear(2048+300+300, 1)(all_feat)
  local model = nn.gModule({im_feat, q_feat, choice_feat}, {fc1})
  return model
end


function networks:Bilinear()
  local im_feat = nn.Identity()()
  local q_feat = nn.Identity()()
  local choice_feat = nn.Identity()()
  local q_choice_feat = nn.JoinTable(2)({q_feat, choice_feat})
  local bilinear = nn.Bilinear(2048, 600, 1)({im_feat, q_choice_feat})
  local model = nn.gModule({im_feat, q_feat, choice_feat}, {bilinear})
  return model
end


function networks:MLP(dims)
  local total_dims = 0
  for i = 1, #dims do
    total_dims = total_dims + dims[i]
  end

  local inputs = {}
  local norm_inputs = {}
  for i = 1, #dims do
    inputs[i] = nn.Identity()()
    norm_inputs[i] = nn.Normalize(2)(inputs[i])
  end

  local all_feat = nn.JoinTable(2)(norm_inputs)
  local fc1 = nn.ReLU(true)(nn.BatchNormalization(8192)(
    nn.Linear(total_dims, 8192)
      :init('weight', nninit.xavier, {dist = 'normal', gain = 1})
      (all_feat)))
  local fc2 = nn.Linear(8192, 1)
    :init('weight', nninit.xavier, {dist = 'normal', gain = 1})
    (nn.Dropout(0.5)(fc1))
  local model = nn.gModule(inputs, {fc2})
  
  return model
end


function networks:MLP_qa()
  local q_feat = nn.Identity()()
  local choice_feat = nn.Identity()()
  local all_feat = nn.JoinTable(2)({q_feat, choice_feat})
  local fc1 = nn.ReLU(true)(nn.BatchNormalization(8192)(nn.Linear(300+300, 8192)(all_feat)))
  local fc2 = nn.Linear(8192, 1)(nn.Dropout(0.5)(fc1))
  local model = nn.gModule({q_feat, choice_feat}, {fc2})
  return model
end


return networks