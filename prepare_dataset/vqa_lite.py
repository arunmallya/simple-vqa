import json

filenames = [\
  'data/vqa/Annotations/mscoco_train2014_annotations', \
  'data/vqa/Annotations/mscoco_val2014_annotations', \
  'data/vqa/Annotations/mscoco_trainval2014_annotations', \
]

for filename in filenames:
  with open(filename + '.json') as fin:
    anns = json.load(fin)

  output = {}
  anns_lite = []
  for i, item in enumerate(anns['annotations']):
    print '%d/%d\n' % (i, len(anns['annotations']))
    item.pop('answers', None)
    anns_lite.append(item)

  output['annotations'] = anns_lite

  with open(filename + '_lite.json', 'w') as fout:
    json.dump(output, fout)
