local json = require 'cjson'
local tds = require 'tds'
local misc_utils = require 'external.th-utils.misc_utils'


local VQAWrapper = torch.class('VQAWrapper')


function VQAWrapper:__init(dataset_root, split)
  local split_name 
  if split == 'train' then
    split_name = 'train2014'
  elseif split == 'val' then
    split_name = 'val2014'
  elseif split == 'trainval' then
    split_name = 'trainval2014'
  elseif split == 'test-dev' then
    split_name = 'test-dev2015'
  elseif split == 'test-final' then
    split_name = 'test2015'
  else
    error('Split = ' .. split .. ' not recognized')
  end
  self.split = split

  -- Load the annotations.
  self.questions, self.answers = self:loadData(dataset_root, split_name)

  self.num_questions = #self.questions
  print('Loaded ' .. self.num_questions .. ' questions from the ' .. split .. ' split')
end


function VQAWrapper:loadData(dataset_root, split_name)
  if split_name == 'trainval2014' then
    train_questions, train_answers = self:loadData(dataset_root, 'train2014')
    val_questions, val_answers = self:loadData(dataset_root, 'val2014')

    -- Append val to train.
    for i = 1, #val_questions do
      train_questions[#train_questions + 1] = val_questions[i]
    end
    for i = 1, #val_answers do
      train_answers[#train_answers + 1] = val_answers[i]
    end

    return train_questions, train_answers
  else
    local questions_file = dataset_root .. '/Questions/MultipleChoice_mscoco_' .. split_name .. '_questions.json'
    print('Loading ' .. questions_file)
    questions = misc_utils.read_json(questions_file)
    -- Send table to cdata to overcome luajit memory restriction of 2GB.
    questions = tds.Hash(questions)
    -- :|
    collectgarbage()
    collectgarbage()
    collectgarbage()

    if split_name == 'test2015' or split_name == 'test-dev2015' then
      answers = {}
    else 
      local answers_file = dataset_root .. '/Annotations/mscoco_' .. split_name .. '_annotations_lite.json'
      print('Loading ' .. answers_file)
      answers = misc_utils.read_json(answers_file)
      -- Send table to cdata to overcome luajit memory restriction of 2GB.
      answers = tds.Hash(answers)
    end
    -- :|
    collectgarbage()
    collectgarbage()
    collectgarbage()

    return questions.questions, answers.annotations
  end
end

function VQAWrapper:getNumQuestions()
  return self.num_questions
end


function VQAWrapper:getQuestion(i)
  assert(i > 0 and i <= self.num_questions, 
    'Question index out of range. Index ' .. i .. ' not in [1, ' .. 
    self.num_questions .. ']')

  local output = misc_utils.deepcopy(self.questions[i])
  if self.split ~= 'test-dev' and self.split ~= 'test-final' then
    local answer = self.answers[i]
    assert(answer.question_id == output.question_id)
    assert(answer.image_id == output.image_id)
    output.answer = answer.multiple_choice_answer
    output.answer_id_in_choices = misc_utils.find_key_of_value_in_table(
      output.multiple_choices, output.answer)
    assert(output.answer_id_in_choices ~= nil)
  end

  return output
end


