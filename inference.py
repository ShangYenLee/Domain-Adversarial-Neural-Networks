import os
import csv
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.transforms import *
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--test_loc',type=str,default='../hw2_data/digits/svhn/val')
parser.add_argument('--out_path',type=str,default='../pred.csv')
parser.add_argument('--checkpoint',type=str,default='./checkpoint/ep260_svhn_acc=0.41745.pt')
parser.add_argument('--num_class',type=int,default=10,help='number of class')
opt = parser.parse_args()

def inference(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    result_file = open(os.path.join(opt.out_path),mode='w',newline='')
    writer = csv.writer(result_file)
    writer.writerow(['image_name','label'])
    model = torch.load(opt.checkpoint)
    model.to(device)

    inference_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
            ])

    with torch.no_grad():
        model.eval()
        file_list = [file for file in os.listdir(opt.test_loc) if file.endswith('.png')]	
        for file_name in tqdm(file_list,desc='Testing'):
            image_loc = os.path.join(opt.test_loc,file_name)
            image = Image.open(image_loc)
            image = inference_transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            class_pred, _ = model(image,0)
            class_pred = class_pred.argmax(dim=1).item()
            writer.writerow([file_name,class_pred])
        result_file.close()

if __name__ == '__main__':
    inference(opt)