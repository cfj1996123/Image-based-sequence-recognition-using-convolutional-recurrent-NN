import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms

from torchvision.transforms import Compose
from warpctc_pytorch import CTCLoss
import cv2
import numpy as np
from tqdm import tqdm
import os

from model import crnn
from dataset import *


## checkpoint save dir
ckpt_dir = 'checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)

## path to load previous checkpoint
restore_from_checkpoint = ''

## train or eval
is_train = True

## gpu info
gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cuda = True if gpu is not '' else False

## synthetic dataset
syn_width = 256
syn_height = 32
syn_num_training = 10000
syn_num_test = 32
## preprocessing
resize_width = 256
resize_height = 32
gauss_mean = 50.0
gauss_std = 25.0
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

## hyperparameters
NUM_epochs = 100
snapshot_interval = 10
eval_interval = 1

base_lr = 1e-3
batch_size = 64
hidden_dim = 128
###


transform = Compose([resize(size=(resize_width, resize_height)),
                     #gaussian_noise(gauss_mean, gauss_std),
                     ToTensor(),
                     Normalize(normalize_mean, normalize_std)
                     ])


# trainset = synthetic_train(height=syn_height, width=syn_width, num_instances=syn_num_training, transform=transform)
# testset = synthetic_train(height=syn_height, width=syn_width, num_instances=syn_num_test, transform=transform)

trainset = coco_train(root_dir='cropped_COCO',annotation='desc.json', transform=transform)
testset = coco_test(root_dir='cropped_COCO',annotation='desc.json', transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=train_data_collate)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=test_data_collate)


# for debug use
# while True:
#     sample = trainset.__getitem__(np.random.choice(range(trainset.__len__())))
#     cv2.imshow(sample['text'], inv_transform(sample['image']))
#     key = cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     if key == 27:
#         break


net = crnn(hid_dim=hidden_dim, chardict=trainset.chardict)
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

        pred_lens = Variable(torch.Tensor( [preds.size(0)] * preds.size(1) ).int())
        loss = loss_function(preds, labels, pred_lens, label_lens) / batch_size
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), 10.0)

        mean_loss.append(loss.data[0])
        optimizer.step()

        ## set description
        description = 'epoch:{}, iteration:{}, current loss:{}, mean loss:{}'.format(epoch, iter, loss.data[0], np.mean(mean_loss))
        iterator.set_description(description)


    epoch += 1
    if epoch % snapshot_interval == 0 or epoch == NUM_epochs:
        ckpt_path = join(ckpt_dir, 'crnn_ckpt_epoch{}'.format(epoch))
        torch.save( {'epoch': epoch, 'loss': loss.data[0],
                     'model_params': net.state_dict(),
                     'optim_params': optimizer.state_dict() },
                    ckpt_path
                    )

    # if epoch % eval_interval == 0 or epoch == NUM_epochs:
    #     net.eval()
    #     for test_batch in testloader:
    #         #test_batch = next(iter(testloader))
    #         test_batch_size = test_batch['img'].size(0)
    #         with torch.no_grad():
    #             imgs = test_batch['img']
    #             if cuda:
    #                 imgs = imgs.cuda()
    #
    #             preds = net(imgs).cpu()
    #         preds = torch.argmax(preds, dim=2)
    #         preds = preds.permute(1, 0)
    #
    #         for i in range(test_batch_size):
    #             pred_label = net.seq_to_text(preds[i].tolist())
    #             true_label = test_batch['text'][i]
    #             print('true: {}, pred: {}'.format(true_label, pred_label))
    #
    #     net.train()

