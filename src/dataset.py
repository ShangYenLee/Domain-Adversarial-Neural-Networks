import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from src.transforms import *

class digitDataset(Dataset):
	def __init__(self, idx, opt, type, transform=None):
		super(digitDataset,self).__init__()
		self.root = opt.train_root
		self.file = opt.train_file
		self.info = pd.read_csv(os.path.join(self.root,self.file,f'{type}.csv'))
		self.filename = self.info['image_name']
		self.label = self.info['label']
		self.transform = transform

	def __len__(self):
		return len(self.label)

	def __getitem__(self,i):
		label = int(self.label[i])
		filename = self.filename[i]
		data = {
			'image': os.path.join(self.root,self.file,'data',filename),
			'label': label
		}

		if self.transform is not None:
			data = self.transform(data)

		return data

def get_digits_dataset(opt, type):
	idx = range(38464)

	train_transform = transforms.Compose([
		LoadImg(keys=['image']),
		totensor(keys=['image']),
		ImgNormal(keys=['image']),
		])

	train_set = digitDataset(idx,opt,type,train_transform)

	return train_set