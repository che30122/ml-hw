from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
#import os
from pathlib import Path
from PIL import Image


class myDataset(Dataset):
	def __init__(self,root_dir,transform=None,train=0):
		self.root_dir=Path(root_dir)
		if(train==0):
			csv_path=self.root_dir / 'train.csv'
		else:
			csv_path=self.root_dir / 'dev.csv'
		csv=pd.read_csv(csv_path,encoding="utf8")
		self.x=list()
		self.y=list()
		self.transform=transform
		#j=0
		for i in csv.iloc[:,0]:
			self.x.append(i)
			#self.y.append(csv.iloc[j,1])
			#j=j+1
		for i in csv.iloc[:,1]:
			if(i=="A"):
				self.y.append(0)
			elif(i=="B"):
				self.y.append(1)
			else:
				self.y.append(2)
		self.y=np.asarray(self.y)
		self.y=torch.from_numpy(self.y)

	def __len__(self):
		return len(self.y)

		
	def __getitem__(self,index):
		image_path=self.root_dir / self.x[index]
		image=Image.open(image_path).convert("RGB")
		if(self.transform):
			image=self.transform(image)

		return image,self.y[index]
'''

csv=pd.read_csv("/home/che/ml_hw2_image/train.csv",encoding="utf8")
#print(csv.iloc[:,0])
x=[]
y=[]
j=0;
for i in csv.iloc[:,0]:
	x.append(i)
	y.append(csv.iloc[j,1])
	j=j+1

print(x)
print(y)
		'''
