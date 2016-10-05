-- Modified by Arun Mallya, amallya2@illinois.edu
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.

--[[ Usage:
th prepare_dataset/extract_imagenet_features.lua data/models/resnet-101.t7 60 \
~/Desktop/datasets/COCO/ val data/  
th prepare_dataset/extract_imagenet_features.lua data/models/resnet-101.t7 60 \
~/Desktop/datasets/COCO/ train data/  
--]]
      
require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'hdf5'
local t = require 'external/fb.resnet.torch/datasets/transforms'
coco = require 'coco'

if #arg < 5 then
  io.stderr:write('Usage (Batch mode)      : th extract-features.lua [MODEL] [BATCH_SIZE] [COCO_ROOT_DIRECTORY] [COCO_SPLIT] [OUTPUT_DIRECTORY]\n')
  os.exit(1)
end
local model_loc = arg[1]
if not paths.filep(model_loc) then
  io.stderr:write('Model file not found at ' .. model_loc .. '\n')
  os.exit(1)
end
local batch_size = tonumber(arg[2])
local coco_root = arg[3]
local coco_split = arg[4]
local h5_dir = arg[5]


-- Initialize COCO api.
local data_type, ann_filename
if coco_split == 'train' then
  data_type = 'train2014'
  ann_filename = 'instances_' .. data_type
elseif coco_split == 'val' then
  data_type = 'val2014'
  ann_filename = 'instances_' .. data_type
elseif coco_split == 'test' then
  data_type = 'test2015'
  ann_filename = 'image_info_' .. data_type
end
local ann_file = coco_root .. '/annotations/' .. ann_filename .. '.json'
local coco_api = coco.CocoApi(ann_file)
local img_ids = coco_api:getImgIds()


-- Get the list of files corresponding to ids.
local list_of_filenames = {}
for i = 1, img_ids:size(1) do 
  img = coco_api:loadImgs(img_ids[i])[1]
  list_of_filenames[i] = coco_root .. '/images/' .. data_type .. '/' .. img.file_name
end
local number_of_files = #list_of_filenames
if batch_size > number_of_files then batch_size = number_of_files end


-- Load the model.
local model = torch.load(model_loc)
-- Remove the fully connected layer.
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)
-- Evaluate mode.
model:evaluate()
-- The model was trained with this input normalization.
local meanstd = {
  mean = { 0.485, 0.456, 0.406 },
  std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
  -- t.Scale(256),
  t.ColorNormalize(meanstd),
  -- t.CenterCrop(224),
}

-- Create the hdf5 file.
local h5_filename = h5_dir .. '/' .. coco_split .. '_features.h5'
local h5_file = hdf5.open(h5_filename, 'w')
local h5_options = hdf5.DataSetOptions()
h5_options:setChunked(1, 2048)
local first_write = true

for i = 1, number_of_files, batch_size do
  print(i .. '/' .. number_of_files)
  local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform 
  local ids_batch = torch.IntTensor(1, batch_size)

  -- Preprocess the images for the batch.
  local image_count = 0
  for j = 1, batch_size do 
    img_name = list_of_filenames[i+j-1] 
    if img_name  ~= nil then
      image_count = image_count + 1
      local img = image.load(img_name, 3, 'float')
      -- Scale the image to 224 x 224.
      img = image.scale(img, 224, 224)
      img = transform(img)
      img_batch[{j, {}, {}, {} }] = img
    end
  end

  -- If this is last batch it may not be the same size, so check that.
  if image_count ~= batch_size then
    img_batch = img_batch[{{1, image_count}, {}, {}, {} } ]
  end

  -- Get list of ids for selected images.
  for j = 1, image_count do
    ids_batch[{1, j}] = img_ids[i+j-1]
  end
  ids_batch = ids_batch[{{}, {1, image_count}}]

  -- Get the output of the layer before the (removed) fully connected layer.
  local output = model:forward(img_batch:cuda()):squeeze(1)

  -- This is necesary because the model outputs different dimension based on size of input.
  if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 
  output = output:float()
  ids_batch = ids_batch:t()

  assert(ids_batch:size(1) == output:size(1))

  -- Write the batched data to file.
  if first_write then
    h5_file:write('data/features', output, h5_options)
    h5_file:write('data/imageid', ids_batch, h5_options)
    h5_file:close()
    first_write = false
    h5_file = hdf5.open(h5_filename, 'r+')
  else
    h5_file:append('data/features', output, h5_options)
    h5_file:append('data/imageid', ids_batch, h5_options)
  end
end

h5_file:close()
print('Saved features to ' .. h5_filename)
