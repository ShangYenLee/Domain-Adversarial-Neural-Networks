import torch
import numpy as np
from tqdm import tqdm

def dann_train_step(epoch, opt, model, sorce_loader, target_loader, criterion, optimizer, device):
    start_steps = opt.ep * len(sorce_loader)
    total_steps = opt.ep * len(sorce_loader)
    model.train()
    train_bar = tqdm(zip(sorce_loader, target_loader), total=min(len(sorce_loader), len(target_loader)), desc=f'Training {epoch:0>3}')
    num, correct = 0, 0
    for i, data in enumerate(train_bar):
        p = float(i + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10 * p)) - 1
        s_img = data[0]['image'].to(device)
        s_lab = data[0]['label'].to(device)
        t_img = data[1]['image'].to(device)
        t_lab = data[1]['label'].to(device) 
        size = min(s_img.shape[0],t_img.shape[0])
        s_img, s_lab = s_img[0:size,:,:,:], s_lab[0:size]
        t_img = t_img[0:size,:,:,:]
        sorce_lab = torch.zeros(s_img.shape[0]).long().to(device)
        target_lab = torch.ones(s_img.shape[0]).long().to(device)
    
        s_class_pred, s_domain_pred = model(s_img,alpha=constant)
        class_loss = criterion(s_class_pred,s_lab)
        s_domain_loss = criterion(s_domain_pred, sorce_lab)

        t_class_pred, t_domain_pred = model(t_img,alpha=constant)
        t_domain_loss = criterion(t_domain_pred,target_lab)

        domain_loss = s_domain_loss + t_domain_loss

        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()
        num += s_img.shape[0]
        correct += (t_class_pred.argmax(dim=1) == t_lab).sum()
        acc = correct/num
        train_bar.set_postfix({
                'Loss': loss.item(), f'Acc': acc.item()
            }) 
    train_bar.close()
    return constant

def train_step(epoch, model, train_loader, criterion, optimizer, constant, device):
    num, correct = 0, 0
    train_bar = tqdm(train_loader, desc=f'Training {epoch:0>3}')
    model.train()
    for data in train_bar:
        image = data['image'].to(device)
        label = data['label'].to(device)
        pred, _ = model(image,alpha=constant)
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += image.shape[0]
        correct += (pred.argmax(dim=1) == label).sum()
        acc = correct/num
        train_bar.set_postfix({
            'Loss': loss.item(), f'Acc': acc.item()
        }) 
    train_bar.close()
    return 0

def val_step(model, val_loader, constant, device):
    model.eval()
    val_bar = tqdm(val_loader, desc=f'Validation')
    with torch.no_grad():
        num, correct = 0, 0
        for data in val_bar:
            image = data['image'].to(device)
            label = data['label'].to(device)
            pred, _ = model(image,alpha=constant)
            num += image.shape[0]
            correct += (pred.argmax(dim=1) == label).sum()
            acc = correct/num
            val_bar.set_postfix({
                f'Acc': acc.item()
            })
        val_bar.close()
    return acc