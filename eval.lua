-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
local json = require 'cjson'
require 'paths'
require 'nngraph'
require 'lfs'

require 'modules.DataLoader'
require 'modules.SigmoidCrossEntropyCriterion'

local opts = require 'opts'
local optim_utils = require 'external.th-utils.optim_utils'
local misc_utils = require 'external.th-utils.misc_utils'

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = opts.parse(arg, 'eval')
print(opt)
print('Looks good? Press Enter to continue.')
io.read()
local checkpoint_path = opt.eval_checkpoint_path
local eval_split = opt.eval_split

local split_name
if eval_split == 'train' then
  split_name = 'train2014'
elseif eval_split == 'val' then
  split_name = 'val2014'
elseif eval_split == 'test-dev' then
  split_name = 'test-dev2015'
elseif eval_split == 'test-final' then
  split_name = 'test2015' 
end
local tag = paths.basename(checkpoint_path, '.t7')

require 'cutorch'
require 'cunn'
cutorch.setDevice(opt.gpu + 1) -- Note +1 because lua is 1-indexed.
dtype = 'torch.CudaTensor'

local checkpoint = torch.load(checkpoint_path)
model = checkpoint.model
-- Overwrite opts with those that the model was trained with.
-- We already have local copies of eval opts, made above.
opt = checkpoint.opt
opt.feat_dim = opt.feat_dim
print(string.format('Using iter = %d', checkpoint.iter))
print('feat_type = ', opt.im_feat_types)
model:cuda()

-------------------------------------------------------------------------------
-- Test function
-------------------------------------------------------------------------------
function eval()
  model:evaluate()
  collectgarbage()
  print('Evaluating on ' .. eval_split .. ' set.')

  -- Set up test data loader.
  local test_loader = DataLoader({
    num_choices_per_q = opt.num_choices_per_q,
    im_feat_types = opt.im_feat_types,
    im_feat_dims = opt.im_feat_dims,
    split = eval_split, 
    vqa_root = opt.data_root .. '/vqa/',
    word2vec_file = opt.data_root .. '/models/GoogleNews-vectors-negative300.bin',
  })
  
  local results, all_preds = {}, {}
  local num_batches, total_samples = 0, 0
  while true do
    if num_batches % 10 == 0 then
      print(test_loader:getIndex() .. '/' .. test_loader:getEpochSize())
    end
    num_batches = num_batches + 1

    -- Get a batch of data.
    local data = {}
    data.im_feats, data.q_feats, data.choice_feats, _, qs = test_loader:oneEpoch(opt.batch_size)
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
    local preds = model:forward(inputs):float()
    local pred_pos_indices = misc_utils.findArgMaxBatch(preds, opt.num_choices_per_q)

    -- Enter prediction into table.
    assert(#qs == pred_pos_indices:size(1))
    for i = 1, #qs do
      local result = {
        question_id = qs[i].question_id, 
        answer = qs[i].multiple_choices[pred_pos_indices[i]]
      }
      local all_pred = {
        question = qs[i].question_id, 
        preds = torch.totable(preds[{
          {(i - 1) * opt.num_choices_per_q + 1, i * opt.num_choices_per_q}
        }])
      }
      results[#results + 1] = result
      all_preds[#all_preds + 1] = all_pred
    end

    total_samples = total_samples + pred_pos_indices:size(1)
  end
  assert(total_samples == test_loader:getEpochSize())
  collectgarbage()

  return results, all_preds
end

-------------------------------------------------------------------------------
-- Main 
-------------------------------------------------------------------------------
local results, all_preds = eval()

-- Save results to json files.
local vqa_output_filename =  'vqa_MultipleChoice_mscoco_' .. split_name .. '_' .. 
  tag .. '-' .. eval_split .. '_results.json'
local text = json.encode(results)
local fout = io.open(opt.checkpoint_dir .. '/' .. vqa_output_filename, 'w')
fout:write(text)
fout:close()

local full_output_filename =  'vqa_MultipleChoice_mscoco_' .. split_name .. '_' .. 
  tag .. '-' .. eval_split .. '_full-results.json'
local text = json.encode(all_preds)
local fout = io.open(opt.checkpoint_dir .. '/' .. full_output_filename, 'w')
fout:write(text)
fout:close()

-- Create zip file containing vqa results.
if eval_split == 'test-dev' or eval_split == 'test-final' then
  lfs.chdir(opt.checkpoint_dir)
  os.execute('rm results.zip')
  os.execute('zip results.zip ' .. vqa_output_filename)
end
print('Wrote results to ' .. vqa_output_filename)
