import os
import csv
import argparse
import datetime
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from src.dataset import get_digits_dataset
from src.trainer import dann_train_step, train_step, val_step
from src.dann import *
from src.utils import save_topk_ckpt

parser = argparse.ArgumentParser()
parser.add_argument('-bs',type=int,default=100,help='batch size')
parser.add_argument('-ep',type=int,default=300,help='epoch')
parser.add_argument('-lr',type=float,default=3e-4,help='learning rate')
parser.add_argument('--size',type=int,default=28,help='image size')
parser.add_argument('--in_channels',type=int,default=3,help='input channels')
parser.add_argument('--num_classes',type=int,default=10,help='number of class')
parser.add_argument('--topk',type=int,default=5,help='top k checkpoint')
parser.add_argument('--step_size',type=int,default=5,help='step size')
parser.add_argument('--save_dir',type=str,default='./checkpoint')
parser.add_argument('--data_root',type=str,default='../hw2_data/digits/')
parser.add_argument('--target_file',type=str,default='svhn', help='svhn or usps')
parser.add_argument('--val_period', type=int, default=5, help='validation period')
parser.add_argument('--train_mode',type=str,default='dann',help='dann, sorce or target')
parser.add_argument('--num_workers',type=int,default=4,help='num_workers')
opt = parser.parse_args()

def train(opt):
	# initial setting
	ckpt_loc = os.path.join(opt.save_dir,f'{opt.train_mode}_{opt.target_file}_{datetime.today().strftime("%m-%d-%H-%M-%S")}')
	os.makedirs(ckpt_loc,exist_ok=True)
	torch.manual_seed(1)
	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    #model
	model = CNNModel()
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
	scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.ep)

    #get data
	mnistm_train = get_digits_dataset(opt, 'mnistm', 'train')
	target_train = get_digits_dataset(opt, opt.target_file, 'train')
	target_val = get_digits_dataset(opt, opt.target_file,'val')
	mnistm_loader = DataLoader(
		mnistm_train,
		batch_size=opt.bs,
		num_workers=opt.num_workers,
		shuffle=True
	)
	target_loader = DataLoader(
		target_train,
		batch_size=opt.bs,
		num_workers=opt.num_workers,
		shuffle=True
	)
	target_val_loader = DataLoader(
		target_val,
		batch_size=1,
		num_workers=opt.num_workers,
		shuffle=True
	)
	if opt.train_mode == 'sorce':
		train_loader = mnistm_loader
	else:
		train_loader = target_loader
	for epoch in range(1, opt.ep+1):
		if opt.train_mode == 'dann':
			constant = dann_train_step(epoch, opt, model, mnistm_loader, target_loader, criterion, optimizer, device)
		else:
			# train on sorce or target only
			constant = train_step(epoch, model, train_loader, criterion, optimizer, 0, device)
		if epoch % opt.val_period == 0:
			acc = val_step(model, target_val_loader, constant, device)
			save_topk_ckpt(model,ckpt_loc,f'{opt.target_file}_ep{epoch:0>3}_acc={acc.item():.5f}.pt',opt.topk+1)
		scheduler.step()

if __name__ == '__main__':
	train(opt)