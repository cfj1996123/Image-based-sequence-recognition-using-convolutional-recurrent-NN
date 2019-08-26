import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
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


class crnn(nn.Module):
    def __init__(self, hid_dim, char_dim):
        super().__init__()
        ## (batch_size,3,a,b) --> (batch_size,512,a/32,b/32)
        self.vgg16 = models.vgg16(pretrained=True).features
        ## (seq_len,batch_size,dim1) --> (seq_len,batch_size,2*hid_dim)
        self.bilstm = nn.LSTM(input_size=512 * 1, hidden_size=hid_dim, batch_first=False, num_layers=2,
                              dropout=0, bidirectional=True)
        self.linear1 = nn.Linear(in_features=2 * hid_dim, out_features=char_dim+1, bias=True)  # add a non-character class

    def forward(self, x):

        x = self.vgg16(x)
        x = x.permute(3, 0, 1, 2)
        size = x.size()
        z = x.view(size[0], size[1], size[2] * size[3])
        z = self.bilstm(z)[0]
        z = self.linear1(z)  # out: (seq_len,batch_size,char_dim+1)

        return z


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

## hyperparameters
gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cuda = True if gpu is not '' else False

NUM_epochs = 100
checkpoint_interval = 5
base_lr = 1e-3
batch_size = 32
hidden_dim = 128
###


transform = Compose([ resize(size=(224,32)) ])

trainset = coco_train(root_dir='cropped_COCO',annotation='desc.json', transform=transform)


# data = trainset.__getitem__(0)
# print(data['image'].shape)
# print(data['image'].transpose(2,0,1).shape)


trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1, collate_fn=data_collate)

#iterator = tqdm(trainloader)
num_chars = len(trainset.mydict)
net = crnn(hid_dim=hidden_dim, char_dim=num_chars)
net.train()

if cuda:
    net = net.cuda()



optimizer = optim.Adam(net.parameters(), lr = base_lr, weight_decay=0.0001)
loss_function = CTCLoss()

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
        ckpt_path = 'crnn_ckpt_epoch{}'.format(epoch)
        torch.save( {'epoch': epoch, 'loss': loss.data[0],
                     'model_params': net.state_dict(),
                     'optim_params': optimizer.state_dict() },
                    ckpt_path
                    )


