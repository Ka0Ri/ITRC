import yaml
from glob import glob

train_images = glob('dataset/hardhat/train/images/*jpg')
test_images = glob('dataset/hardhat/test/images/*jpg')

print(f'total number of train images: {len(train_images)}')
print(f'total number of train images: {len(test_images)}')

with open('dataset/hardhat/train.txt', 'w') as f:
  f.write('\n'.join(train_images) + '\n')

with open('dataset/hardhat/test.txt', 'w') as f:
  f.write('\n'.join(test_images) + '\n')

with open('dataset/hardhat/data.yaml', 'r') as f:
  data = yaml.load(f, Loader=yaml.FullLoader)

print(data)

data['train'] = 'dataset/hardhat/train.txt'
data['val'] = 'dataset/hardhat/test.txt'

with open('dataset/hardhat/data.yaml', 'w') as f:
  yaml.dump(data, f)

print(data)