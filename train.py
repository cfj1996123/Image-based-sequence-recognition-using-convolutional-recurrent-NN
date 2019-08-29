import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from torchvision.transforms import Compose
from warpctc_pytorch import CTCLoss

import json
from os.path import join
import cv2
import numpy as np
from tqdm import tqdm
import os

from model import crnn


class coco_train(Dataset):
    def __init__(self, root_dir, annotation, transform=None):
        with open(join(root_dir, annotation)) as f:
            myanno = json.load(f)

        self.mydict = myanno['abc']  # type: str
        self.root_dir = root_dir
        self.annotation = myanno['train']  # list of dicts {'text':..., 'name':...}
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_name = self.annotation[idx]['name']
        text = self.annotation[idx]['text']

        img = cv2.imread(join(self.root_dir, img_name))  # shape: row * col * channels

        seq = self.text_to_seq(text)
        sample = {'text': seq, 'image': img}
        if self.transform:
            sample = self.transform(sample) # shape: predefined_height * predefined_width * channels

        return sample

    def text_to_seq(self, text):
        return [self.mydict.find(char)+1 for char in text if self.mydict.find(char)>=0]   # 0 means non-character


# data transform
class resize:
    def __init__(self, size=(224, 32)):
        self.size = size

    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.size)
        return sample


# strategy to combine a batch of data
def data_collate(batch):
    img = []
    seq = []
    seq_len = []
    for sample in batch:
        img.append(torch.from_numpy(sample['image'].transpose((2, 0, 1))).float())
        seq.extend(sample['text'])
        seq_len.append(len(sample['text']))

    img = torch.stack(img)
    seq = torch.Tensor(seq).int()
    seq_len = torch.Tensor(seq_len).int()

    batch = {"img": img, "seq": seq, "seq_len": seq_len}
    return batch


## other info
ckpt_dir = 'checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)
restore_from_checkpoint = 'checkpoints/crnn_ckpt_epoch60'
is_train = True

## hyperparameters
gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cuda = True if gpu is not '' else False

NUM_epochs = 500
checkpoint_interval = 20
base_lr = 1e-3
batch_size = 32
hidden_dim = 128
###


transform = Compose([ resize(size=(224,32)) ])

trainset = coco_train(root_dir='cropped_COCO',annotation='desc.json', transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1, collate_fn=data_collate)


num_chars = len(trainset.mydict)
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
    mean_loss = [ ckpt['loss'] ]
else:
    epoch = 0
    mean_loss = []

while epoch < NUM_epochs:
    iterator = tqdm(trainloader)
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
    if epoch % checkpoint_interval == 0 or epoch == NUM_epochs:
        ckpt_path = join(ckpt_dir, 'crnn_ckpt_epoch{}'.format(epoch))
        torch.save( {'epoch': epoch, 'loss': loss.data[0],
                     'model_params': net.state_dict(),
                     'optim_params': optimizer.state_dict() },
                    ckpt_path
                    )


