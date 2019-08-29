import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision.transforms import Compose
from warpctc_pytorch import CTCLoss


import numpy as np
from tqdm import tqdm
import os

from model import crnn
from dataset import *


## other info
ckpt_dir = 'checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)
restore_from_checkpoint = ''
is_train = True

## hyperparameters
gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cuda = True if gpu is not '' else False

NUM_epochs = 500
snapshot_interval = 10
eval_interval = 10

base_lr = 1e-3
batch_size = 32
hidden_dim = 256
###


transform = Compose([ resize(size=(224,32)) ])

trainset = coco_train(root_dir='cropped_COCO',annotation='desc.json', transform=transform)
testset = coco_test(root_dir='cropped_COCO',annotation='desc.json', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=train_data_collate)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=test_data_collate)



net = crnn(hid_dim=hidden_dim, chardict=trainset.mydict)
if is_train:
    net.train()
else:
    net.eval()
if cuda:
    net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr = base_lr, weight_decay=0.0001)
loss_function = CTCLoss()

## restore from checkpoint
if restore_from_checkpoint != '':
    ckpt = torch.load(restore_from_checkpoint)
    net.load_state_dict(ckpt['model_params'])
    optimizer.load_state_dict(ckpt['optim_params'])
    epoch = ckpt['epoch']
else:
    epoch = 0



# net.eval()
# test_batch = next(iter(testloader))
# test_batch_size = test_batch['img'].size(0)
#
# preds = net(test_batch['img'].cuda()).cpu()
# preds = torch.argmax(preds, dim=2)
# preds = preds.permute(1,0)
# print(preds.size())
# print(preds)
# for i in range(test_batch_size):
#     pred_label = net.seq_to_text( preds[i].tolist() )
#     true_label = test_batch['seq'][i]
#     print( 'true: {}, pred: {}'.format(true_label, pred_label) )



while epoch < NUM_epochs:
    iterator = tqdm(trainloader)
    mean_loss = []
    for iter, batch in enumerate(iterator):
        optimizer.zero_grad()
        imgs = Variable(batch['img'])
        labels = Variable(batch['seq'])
        label_lens = Variable(batch['seq_len'].int())
        if cuda:
            imgs = imgs.cuda()

        preds = net(imgs).cpu()
        #print(preds.size())
        pred_lens = Variable(torch.Tensor( [preds.size(0)] * preds.size(1) ).int())
        loss = loss_function(preds, labels, pred_lens, label_lens) / batch_size
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), 1.0)

        mean_loss.append(loss.data[0])
        optimizer.step()

        ## set description
        description = 'epoch:{}, iteration:{}, current loss:{}, mean loss:{}'.format(epoch, iter, loss.data[0], np.mean(mean_loss))
        iterator.set_description(description)
        #print(description)

    epoch += 1
    if epoch % snapshot_interval == 0 or epoch == NUM_epochs:
        ckpt_path = join(ckpt_dir, 'crnn_ckpt_epoch{}'.format(epoch))
        torch.save( {'epoch': epoch, 'loss': loss.data[0],
                     'model_params': net.state_dict(),
                     'optim_params': optimizer.state_dict() },
                    ckpt_path
                    )

    if epoch % eval_interval == 0 or epoch == NUM_epochs:
        net.eval()
        test_batch = next(iter(testloader))
        test_batch_size = test_batch['img'].size(0)
        with torch.no_grad():
            preds = net(test_batch['img'].cuda()).cpu()
        preds = torch.argmax(preds, dim=2)
        preds = preds.permute(1, 0)
        print(preds.size())
        print(preds)
        for i in range(test_batch_size):
            pred_label = net.seq_to_text(preds[i].tolist())
            true_label = test_batch['seq'][i]
            print('true: {}, pred: {}'.format(true_label, pred_label))

        net.train()

