local opts = {}

function opts.train()
  local cmd = torch.CmdLine()

  -- Data location.
  cmd:option('-data_root', 'data/', 'Relative location of dataset')
  cmd:option('-checkpoint_start_from', '',
    'Load model from a checkpoint instead of random initialization.')
  
  -- Core Net settings.
  cmd:option('-net_type', 'MLP', 'MLP/MLP_red/Bilinear/Linear')
  cmd:option('-net_inputs', 'iqa', 'iqa/qa')
  cmd:option('-im_feat_types', 'imagenet', 'imagenet')
  cmd:option('-im_feat_dims', '2048', '2048')
  cmd:option('-train_split', 'train', 'train/trainval')
  
  -- Train settings.
  cmd:option('-batch_size', 512, 'Number of images in a batch')
  cmd:option('-balance_data', false, 'Equal number of positive and negative in a batch')
  cmd:option('-num_choices_per_q', 18, 'Number of choices per question')

  -- Loss function weights.
  cmd:option('-weight_decay', 1e-6, 'L2 weight decay penalty strength')

  -- Optimization.
  cmd:option('-base_lr', 0.01, 'learning rate to use')
  cmd:option('-decay_type', 'step', 'fixed/step/exp')
  cmd:option('-gamma', 0.9, 'lr decay param')
  cmd:option('-step', 100000, 'lr decay param')
  
  cmd:option('-update', 'sgdm', 'sgdm/adam')
  cmd:option('-optim_beta1', 0.9, 'beta1 for adam')
  cmd:option('-optim_beta2', 0.999, 'beta2 for adam')
  cmd:option('-optim_epsilon', 1e-8, 'epsilon for smoothing')

  cmd:option('-max_iters', -1, 'Number of iterations to run; -1 to run forever')
  
  -- Model checkpointing.
  cmd:option('-save_checkpoint_every', 20000,
    'How often to save model checkpoints')
  cmd:option('-checkpoint_dir', 'checkpoint/',
    'Name of the checkpoint file to use')

  -- Visualization.
  cmd:option('-progress_dump_every', 100,
    'Every how many iterations do we write a progress report to vis/out ?. 0 = disable.')
  cmd:option('-losses_log_every', 10,
    'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

  -- Misc.
  cmd:option('-id', '',
    'an id identifying this run/job; useful for cross-validation')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-eval_first_iteration', 1,
    'evaluate on first iteration? 1 = do, 0 = dont.')

  return cmd
end


function opts.eval()
  local cmd = torch.CmdLine()

  -- Eval opts.
  cmd:option('-eval_checkpoint_path', '', 'For eval time usage.')
  cmd:option('-eval_split', '', 'train/val/test-dev/test-final')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
  
  return cmd
end


function opts.parse(arg, opt_type)
  local cmd
  if opt_type == 'train' then
    cmd = opts.train()
  elseif opt_type == 'eval' then
    cmd = opts.eval()
  end
  cmd:text()

  local opt = cmd:parse(arg or {})
  return opt
end

return opts
