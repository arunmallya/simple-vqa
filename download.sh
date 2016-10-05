# Download the pretrained models.
mkdir data/
mkdir data/models
cd data/models

echo 'Downloading ImageNet resnet'
wget https://d2j0dndfm35trm.cloudfront.net/resnet-152.t7

cd ../..


# Word2Vec
echo "-------------"
echo "-- Download word2vec embeddings from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit"
echo "-------------"