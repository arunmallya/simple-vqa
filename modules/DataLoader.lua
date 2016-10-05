require 'hdf5'
require 'modules.VQAWrapper'
require 'torchx'
require 'external.th-utils.Word2Vec'


local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt)
  self.num_choices_per_q = opt.num_choices_per_q
  self.split = opt.split

  self.im_feat_types = opt.im_feat_types
  self.im_feat_dims = opt.im_feat_dims
  local h5_files, h5_handles, img_id_to_h5_index = {}, {}, {}
  for j = 1, #self.im_feat_types do
    local h5_split = opt.split
    if opt.split == 'test-dev' or opt.split == 'test-final' then
      h5_split = 'test'
    end
    h5_files[j] = 'data/feats/' .. self.im_feat_types[j] .. '_' .. h5_split .. '_features.h5'
    h5_handles[j] = hdf5.open(h5_files[j], 'r')
    img_id_to_h5_index[j] = h5_handles[j]:read('data/imageid'):all()
  end
  self.h5_files = h5_files
  self.h5_handles = h5_handles
  self.img_id_to_h5_index = img_id_to_h5_index

  self.vqa_root = opt.vqa_root
  self.word2vec_file = opt.word2vec_file or nil
  self.w2v = opt.word2vec or nil

  self.vqa = VQAWrapper(self.vqa_root, self.split)
  self.num_questions = self.vqa:getNumQuestions()
  self:resetIndex()
  self.epoch_over = false  

  if self.w2v == nil then
    print('Initializing word2vec')
    self.w2v = Word2Vec(self.word2vec_file)
  end
  print('DataLoader initialized')

  -- :|
  collectgarbage()
  collectgarbage()
  collectgarbage()
end


function DataLoader:resetIndex()
  self.question_order = torch.randperm(self.num_questions)
  self.index = 1
end

function DataLoader:getIndex()
  return self.index
end


function DataLoader:getEpochSize()
  return self.num_questions
end


function DataLoader:getBatch(num_samples, opt)
  assert(self.split ~= 'test-dev' and self.split ~= 'test-final', 
    'This can only be called on the train/val set.')
  local half_and_half = opt.half_and_half

  -- Initialize outputs.
  local im_feats = {}
  for j = 1, #self.im_feat_types do
    im_feats[j] = torch.Tensor(num_samples, self.im_feat_dims[j])
  end
  local q_feats = torch.Tensor(num_samples, 300)
  local choice_feats = torch.Tensor(num_samples, 300)
  local labels = torch.Tensor(num_samples)
  local questions = {}

  -- Gather samples.
  for i = 1, num_samples do
    if self.index > self.num_questions then
      self:resetIndex()
    end

    -- Get a question from the VQA annotations.
    local question = self.vqa:getQuestion(self.question_order[self.index])

    -- Load each of the image feats.
    for j = 1, #self.im_feat_types do
      local h5_index = torch.find(self.img_id_to_h5_index[j], question.image_id)[1]
      local im_feat = self.h5_handles[j]:read('data/features')
        :partial({h5_index, h5_index}, {1,self.im_feat_dims[j]})
      local im_id = self.h5_handles[j]:read('data/imageid')
        :partial({h5_index, h5_index}, {1,1})
      assert(question.image_id == im_id[{1, 1}])
      -- Insert feats into outputs.
      im_feats[j][{i, {}}] = im_feat
    end
    
    -- Choose a choice to return.
    local label, choice
    if half_and_half then
      -- If half_and_half is true, choose the correct answer with prob. 0.5
      if torch.uniform() < 0.5 then
        choice = question.answer
      else
        -- Choose one of the negatives randomly.
        local choice_index = torch.random(1, #(question.multiple_choices) - 1)
        if choice_index >= question.answer_id_in_choices then
        -- Skip over the correct choice.
          choice_index = choice_index + 1
        end
        choice = question.multiple_choices[choice_index]
        -- Sanity check.
        assert(choice ~= question.answer)
      end
    else
      -- Just choose a choice at random.
      choice = question.multiple_choices[
        torch.random(1, #(question.multiple_choices))]
    end

    if choice == question.answer then
      label = 1
    else 
      label = 0
    end

    -- Insert feats into outputs.
    q_feats[{i, {}}] = self.w2v:encode(question.question, {average = true})
    choice_feats[{i, {}}] = self.w2v:encode(choice, {average = true})
    labels[i] = label
    question.choice = choice
    questions[i] = question

    self.index = self.index + 1
  end

  return im_feats, q_feats, choice_feats, labels, questions
end


function DataLoader:getFullBatch(num_qs)
  assert(self.split ~= 'test-dev' and self.split ~= 'test-final', 
    'This can only be called on the train/val set.')
  
  -- Initialize outputs.
  local im_feats = torch.Tensor(num_qs*self.num_choices_per_q, self.im_feat_dim)
  local q_feats = torch.Tensor(num_qs*self.num_choices_per_q, 300)
  local choice_feats = torch.Tensor(num_qs*self.num_choices_per_q, 300)
  local labels = torch.Tensor(num_qs*self.num_choices_per_q)
  local questions = {}

  -- Gather samples.
  for i = 1, num_qs do
    if self.index > self.num_questions then
      self:resetIndex()
    end

    -- Get a question from the VQA annotations.
    local question = self.vqa:getQuestion(self.question_order[self.index])
    local h5_index = torch.find(self.img_id_to_h5_index, question.image_id)[1]
    local im_feat = self.h5_handle:read('data/features')
      :partial({h5_index, h5_index}, {1,self.im_feat_dim})
    local im_id = self.h5_handle:read('data/imageid')
      :partial({h5_index, h5_index}, {1,1})
    assert(question.image_id == im_id[{1, 1}])

    -- Insert into outputs.
    local start_index = (i-1)*self.num_choices_per_q + 1
    local end_index = start_index + self.num_choices_per_q - 1
    im_feats[{{start_index, end_index}, {}}] = im_feat:repeatTensor(self.num_choices_per_q, 1)
    q_feats[{{start_index, end_index}, {}}] = self.w2v:encode(question.question, {average = true}):repeatTensor(self.num_choices_per_q, 1)
    for idx = 1, self.num_choices_per_q do
      choice_feats[{start_index+idx-1, {}}] = self.w2v:encode(question.multiple_choices[idx], {average = true})
      if question.multiple_choices[idx] == question.answer then
        labels[{start_index+idx-1}] = 1
      else
        labels[{start_index+idx-1}] = 0 
      end
    end
    questions[i] = question

    self.index = self.index + 1
  end

  return im_feats, q_feats, choice_feats, labels, questions
end


-- This does not use the random question ordering. Just goes from 1 .. N.
function DataLoader:oneEpoch(num_samples)
  -- In the case that we don't want to iterate over epochs, just return nil
  -- This is useful in val and test set where we want to make just a single 
  -- pass.
  if self.epoch_over or self.index > self.num_questions then
    return nil, nil, nil, nil, nil
  end

  -- Initialize outputs.
  local im_feats = {}
  for j = 1, #self.im_feat_types do
    im_feats[j] = torch.Tensor(num_samples*self.num_choices_per_q, self.im_feat_dims[j])
  end
  local q_feats = torch.Tensor(num_samples*self.num_choices_per_q, 300)
  local choice_feats = torch.Tensor(num_samples*self.num_choices_per_q, 300)
  local pos_index = torch.Tensor(num_samples)
  local questions = {}
  if self.split == 'test-dev' or self.split == 'test-dev' then
    pos_index = nil
  end

  -- Gather samples.
  local count
  for i = 1, num_samples do
    if self.index > self.num_questions then
      self.epoch_over = true
      for j = 1, #self.im_feat_types do
        im_feats[j] = im_feats[j][{{1, count}, {}}]
      end
      q_feats = q_feats[{{1, count}, {}}]
      choice_feats = choice_feats[{{1, count}, {}}]
      if self.split ~= 'test-dev' and self.split ~= 'test-final'then
        pos_index = pos_index[{{1, count / self.num_choices_per_q}}]
      end
      return im_feats, q_feats, choice_feats, pos_index, questions
    end

    -- If not end of epoch, insert indices are below.
    local start_index = (i-1)*self.num_choices_per_q + 1
    local end_index = start_index + self.num_choices_per_q - 1

    -- Get a question from the VQA annotations.
    local question = self.vqa:getQuestion(self.index)
    
    -- Get all of the image feats.
    for j = 1, #self.im_feat_types do
      local h5_index = torch.find(self.img_id_to_h5_index[j], question.image_id)[1]
      local im_feat = self.h5_handles[j]:read('data/features')
        :partial({h5_index, h5_index}, {1,self.im_feat_dims[j]})
      local im_id = self.h5_handles[j]:read('data/imageid')
        :partial({h5_index, h5_index}, {1,1})
      assert(question.image_id == im_id[{1, 1}])
      -- Insert image feats into outputs.
      im_feats[j][{{start_index, end_index}, {}}] = im_feat:repeatTensor(self.num_choices_per_q, 1)
    end

    -- Insert word feats into outputs.    
    q_feats[{{start_index, end_index}, {}}] = self.w2v:encode(question.question, {average = true}):repeatTensor(self.num_choices_per_q, 1)
    for idx = 1, self.num_choices_per_q do
      choice_feats[{start_index+idx-1, {}}] = self.w2v:encode(question.multiple_choices[idx], {average = true})
    end
    questions[i] = question
    if self.split ~= 'test-dev' and self.split ~= 'test-final'then
      pos_index[i] = question.answer_id_in_choices
    end

    count = end_index
    self.index = self.index + 1
  end

  return im_feats, q_feats, choice_feats, pos_index, questions
end