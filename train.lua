-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
local json = require 'cjson'
require 'paths'

require 'modules.DataLoader'
require 'modules.SigmoidCrossEntropyCriterion'

local arch = require 'modules.networks'
local opts = require 'opts'
local optim_utils = require 'external.th-utils.optim_utils'
local misc_utils = require 'external.th-utils.misc_utils'

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
-- Get opts.
local opt = opts.parse(arg, 'train')
torch.manualSeed(opt.seed)
paths.mkdir(opt.checkpoint_dir)
opt.checkpoint_prefix = opt.checkpoint_dir .. '/' .. opt.net_type .. '-' .. opt.im_feat_types .. '-' .. opt.train_split
if opt.id ~= '' then
   opt.checkpoint_prefix = opt.checkpoint_prefix .. '-' .. opt.id
end
opt.im_feat_types = misc_utils.split_string(opt.im_feat_types, '_')
opt.im_feat_dims = misc_utils.split_string(opt.im_feat_dims, '_')
for i = 1, #opt.im_feat_dims do
  opt.im_feat_dims[i] = tonumber(opt.im_feat_dims[i])
end
print(opt)
print('Looks good? Press Enter to continue.')
io.read()

-- GPU/cudnn.
torch.setdefaulttensortype('torch.FloatTensor')
require 'cutorch'
require 'cunn'
require 'cudnn'
cutorch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpu + 1) -- Note +1 because lua is 1-indexed.
local dtype = 'torch.CudaTensor'

-- Initialize/Load the model.
local loss_history = {}
local results_history = {}
local iter = 0
local best_val_accuracy = -1

local model
if #(opt.checkpoint_start_from) > 0 then
  local checkpoint = torch.load(opt.checkpoint_start_from)
  iter = checkpoint.iter
  loss_history = checkpoint.loss_history
  results_history = checkpoint.results_history
  best_val_accuracy = checkpoint.best_val_accuracy
  model = checkpoint.model
else
  model = arch:getNetwork(opt.net_type, opt.net_inputs, opt.im_feat_dims)
end
local criterion = nn.SigmoidCrossEntropyCriterion()
criterion.sizeAverage = false
model:cuda()
criterion:cuda()

-- Get the parameters vector.
local params, grad_params = model:getParameters()

-- Initialize the train data loader class.
local train_loader = DataLoader.new({
  num_choices_per_q = opt.num_choices_per_q,  
  im_feat_types = opt.im_feat_types,
  im_feat_dims = opt.im_feat_dims,
  split = opt.train_split, 
  vqa_root = opt.data_root .. '/vqa/',
  word2vec_file = opt.data_root .. '/models/GoogleNews-vectors-negative300.bin',
})

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  grad_params:zero()
  model:training()

  -- Fetch data using the loader.
  local data = {}
  data.im_feats, data.q_feats, data.choice_feats, data.labels, qs = train_loader:getBatch(
    opt.batch_size, {half_and_half = opt.balance_data})

  -- Prepare inputs.
  local inputs 
  if opt.net_inputs == 'iqa' then
    inputs = misc_utils.flatten_table({data.im_feats, data.q_feats, data.choice_feats})
  elseif opt.net_inputs == 'qa' then
    inputs = {data.q_feats, data.choice_feats}
  end
  inputs = misc_utils.dtype(inputs, dtype)
  data.labels = data.labels:type(dtype)

  -- Run the model forward and backward.
  local preds = model:forward(inputs)
  local loss = criterion:forward(preds, data.labels)
  -- loss = loss / (opt.batch_size)

  local dloss_preds = criterion:backward(preds, data.labels)
  -- dloss_preds:div(opt.batch_size)
  model:backward(inputs, dloss_preds)

  -- Apply L2 regularization.
  if opt.weight_decay > 0 then
    grad_params:add(opt.weight_decay, params)
  end

  --++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  -- Visualization/Logging code
  --++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  if opt.losses_log_every > 0 and iter % opt.losses_log_every == 0 then
    loss_history[iter] = loss
  end

  return loss
end

-------------------------------------------------------------------------------
-- Validation 
-------------------------------------------------------------------------------
function evalValidation()
  model:evaluate()
  collectgarbage()
  print('Evaluating on validation set.')

  -- Set up val data loader.
  local val_loader = DataLoader({
    num_choices_per_q = opt.num_choices_per_q,
    im_feat_types = opt.im_feat_types,
    im_feat_dims = opt.im_feat_dims,
    split = 'val', 
    vqa_root = opt.data_root .. '/vqa/',
    word2vec = train_loader.w2v,  -- Reuse word2vec instance of train loader.
  })

  local num_correct, total_samples, num_batches = 0, 0, 0
  while true do
    if num_batches % 100 == 0 then
      print(val_loader:getIndex() .. '/' .. val_loader:getEpochSize())
    end
    num_batches = num_batches + 1

    -- Get a batch of data.
    local data = {}
    data.im_feats, data.q_feats, data.choice_feats, pos_indices, qs = val_loader:oneEpoch(opt.batch_size)
    if data.im_feats == nil then
      break
    end

    -- Prepare inputs.
    local inputs
    if opt.net_inputs == 'iqa' then
      inputs = misc_utils.flatten_table({data.im_feats, data.q_feats, data.choice_feats})
    elseif opt.net_inputs == 'qa' then
      inputs = {data.q_feats, data.choice_feats}
    end
    inputs = misc_utils.dtype(inputs, dtype)

    -- Perform prediction.
    local preds = model:forward(inputs)
    local pred_pos_indices = misc_utils.findArgMaxBatch(preds, opt.num_choices_per_q)

    -- Find accuracy.
    num_correct = num_correct + torch.sum(pos_indices:eq(pred_pos_indices))
    total_samples = total_samples + pos_indices:size(1)
  end
  assert(total_samples == val_loader:getEpochSize())
  collectgarbage()

  local accuracy = num_correct / total_samples
  return accuracy
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}

while true do
  -- Compute loss and gradient.
  local loss = lossFun()

  -- Parameter update  
  local learning_rate = misc_utils.getLearningRate(opt, opt.base_lr, iter)
  if opt.update == 'sgdm' then
    optim_utils.sgdm(params, grad_params, learning_rate, 0.9, optim_state)
  elseif opt.update == 'adam' then
    optim_utils.adam(params, grad_params, learning_rate, opt.optim_beta1, opt.optim_beta2,
      opt.optim_epsilon, optim_state)
  end
  
  -- Print loss and timing/benchmarks.
  if opt.losses_log_every > 0 and iter % opt.losses_log_every == 0 then
    print(string.format('lr: %.3g, iter %d: %g', learning_rate, iter, loss))
  end

  if ((opt.eval_first_iteration == 1 or iter > 0) and iter % opt.save_checkpoint_every == 0) or (iter+1 == opt.max_iters) then
    -- Evaluate validation performance.
    local accuracy
    if opt.train_split ~= 'trainval' then
      accuracy = evalValidation()
      print('Validation Accuracy @iter=' .. iter .. ' = ' .. accuracy)
      results_history[iter] = accuracy
      misc_utils.plotAccuracy(results_history, opt.checkpoint_prefix)
    end

    -- Serialize a json file that has all info except the model.
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.results_history = results_history
    json.encode_number_precision(4) -- Number of sig digits to use in encoding.
    json.encode_sparse_array(true, 2, 10)
    local text = json.encode(checkpoint)
    local fout = io.open(opt.checkpoint_prefix .. '.json', 'w')
    fout:write(text)
    fout:close()
    print('wrote ' .. opt.checkpoint_prefix .. '.json')

    -- Only save t7 checkpoint if there is an improvement in Accuracy.
    if opt.train_split == 'trainval' or accuracy > best_val_accuracy then
      best_val_accuracy = accuracy
      checkpoint.accuracy = accuracy
      model:clearState()
      checkpoint.model = model 
      torch.save(opt.checkpoint_prefix .. '.t7', checkpoint)
      print('wrote ' .. opt.checkpoint_prefix .. '.t7')
    end
  end

  -- Update iteration.
  iter = iter + 1

  -- Collect garbage every so often.
  if iter % 50 == 0 then collectgarbage() end

  -- Stopping criterions.
  if loss0 == nil then loss0 = loss end
  if loss > loss0 * 100 then
    print('Loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end
end