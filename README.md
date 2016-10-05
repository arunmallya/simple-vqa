# simple-vqa
Learns an MLP for VQA

This code implements the VQA MLP basline from [Revisiting Visual Question Answering Baselines](https://arxiv.org/abs/1606.08390) and then some more.

## Some numbers on VQA

| Features/Methods      | VQA Val Accuracy| VQA Test-dev Accuracy |
| ------------- |:---------------:|:-----------------:|
| [MCBP](https://arxiv.org/pdf/1606.01847v2.pdf) | - |  66.4 |
| [Baseline - MLP](https://arxiv.org/abs/1606.08390) | - | 64.9 |
| Imagenet - MLP      | 63.62 | **65.9** |

Readme is a work in progress......

## Installation

The MLP is implemented in [Torch](http://torch.ch/), and depends on the following packages: 
[torch/nn](https://github.com/torch/nn), 
[torch/nngraph](https://github.com/torch/nngraph), 
[torch/cutorch](https://github.com/torch/cutorch), 
[torch/cunn](https://github.com/torch/cunn), 
[torch/image](https://github.com/torch/image), 
[torch/tds](https://github.com/torch/tds), 
[lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson),
[nninit](https://github.com/Kaixhin/nninit),
[torch-word-emb](https://github.com/iamalbert/torch-word-emb),
[torch-hdf5](https://github.com/deepmind/torch-hdf5),
[torchx](https://github.com/nicholas-leonard/torchx)

After installing torch, you can install / update these dependencies by running the following:

```bash
luarocks install nn
luarocks install nngraph
luarocks install image
luarocks install tds

luarocks install cutorch
luarocks install cunn

luarocks install lua-cjson
luarocks install nninit
luarocks install torch-word-emb
luarocks install torchx
```
Install torch-hdf5 by following instructions [here](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)

## Running trained models

### Download this repo
```
git clone --recursive https://github.com/arunmallya/simple-vqa.git
```

### Data Dependencies
* Create a data/ folder and symlink or place the following datasets: vqa -> VQA dataset root, coco -> COCO dataset root (coco is needed only if you plan to extract and use your own features, not required if using cached features below).

* Download the Word2Vec model file from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). This is needed to encode sentences into vectors. Place the .bin file in the data/models folder.

* Download cached features for the VQA dataset splits, unzip them, and place them in data/feats: [features](https://uofi.box.com/s/tewogy0c1de9pq0v26lbum9495az247q) 

* Download [VQA lite annotations](https://uofi.box.com/s/bz0ttp9bowz83xa3i40ieqfbd5r60e04) and place then in data/vqa/Annotations/. These are required because the original VQA annotations do not fit in the 2GB limit of luajit.

* Download MLP models trained on the VQA train set and place them in checkpoint/: [models](https://uofi.box.com/s/wuo9k9j07zq3m72z8nwtq8l0uhs0ji91)

* At this point, your data folder should have models/, feats/, coco/ and vqa/ folders.

### Run Eval

For example, to run the model trained on the VQA train set with Imagenet features, on the VQA val set:
```
th eval.lua -eval_split val \
-eval_checkpoint_path checkpoint/MLP-imagenet-train.t7
```

In general, the command is:
```
th eval.lua -eval_split (train/val/test-dev/test-final) \
-eval_checkpoint_path <model-path>
```

This will dump the results in checkpoint/ as a .json file as well as a results.zip file in case of test-dev and test-final. This results.zip can be uploaded to CodaLab for evaluation.

## Training MLP from scratch

```
th train.lua -im_feat_types imagenet -im_feat_dims 2048
```
