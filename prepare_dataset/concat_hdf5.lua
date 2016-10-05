require 'hdf5'

-- Will vary depending on dataset.
local h5_options = hdf5.DataSetOptions()
h5_options:setChunked(1, 2048)

local data_dir = 'data/feats/'
local feat_type = 'imagenet'
local output_name = feat_type .. '_trainval_features.h5'

local file_list = {
  feat_type .. '_train_features.h5',
  feat_type .. '_val_features.h5'
}
local fields = {
  'data/features',
  'data/imageid'
}

-- Copy the first file.
os.execute('cp ' .. data_dir .. '/' .. file_list[1] .. ' ' .. 
  data_dir .. '/' .. output_name)

-- Append the other files to the first file.
local h5_output = hdf5.open(data_dir .. '/' .. output_name, 'r+')
for i = 2, #file_list do
  local h5_input = hdf5.open(data_dir .. '/' .. file_list[i])
  for j = 1, #fields do
    local values = h5_input:read(fields[j]):all()
    h5_output:append(fields[j], values, h5_options)
  end
  h5_input:close()
end
h5_output:close()
