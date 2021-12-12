import yaml
from glob import glob
import time

train_images = glob('dataset/Mask/train/images/*jpg')
test_images = glob('dataset/Mask/valid/images/*jpg')

print(f'total number of train images: {len(train_images)}')
print(f'total number of train images: {len(test_images)}')

with open('dataset/Mask/train.txt', 'w') as f:
  f.write('\n'.join(train_images) + '\n')

with open('dataset/Mask/test.txt', 'w') as f:
  f.write('\n'.join(test_images) + '\n')

with open('dataset/Mask/data.yaml', 'r') as f:
  data = yaml.load(f, Loader=yaml.FullLoader)

print(data)
time.sleep(10)

data['train'] = 'dataset/Mask/train.txt'
data['val'] = 'dataset/Mask/test.txt'

with open('dataset/Mask/data.yaml', 'w') as f:
  yaml.dump(data, f)

print(data)