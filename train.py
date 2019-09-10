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
from Levenshtein import distance
import argparse
from hparams import create_hparams


def get_dataset(hparams, args):

    transform = Compose([resize(size=(hparams.resize_width, hparams.resize_height)),
                         #gaussian_noise(gauss_mean, gauss_std),
                         ToTensor(),
                         Normalize(hparams.normalize_mean, hparams.normalize_std)
                         ])
    if args.dataset not in ['synthesized', 'coco']:
        raise ValueError('Dataset not supported.')

    if args.dataset == 'synthesized':
        trainset = synthetic_train(height=hparams.syn_height, width=hparams.syn_width, num_instances=hparams.syn_num_training, transform=transform)
        testset = synthetic_train(height=hparams.syn_height, width=hparams.syn_width, num_instances=hparams.syn_num_test, transform=transform)

    if args.dataset == 'coco':
        trainset = coco_train(root_dir='cropped_COCO',annotation='desc.json', transform=transform)
        testset = coco_test(root_dir='cropped_COCO',annotation='desc.json', transform=transform)

    return trainset, testset


def train(hparams, args):

    trainset, testset = get_dataset(hparams, args)
    trainloader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, num_workers=1, collate_fn=train_data_collate)
    testloader = DataLoader(testset, batch_size=hparams.batch_size, shuffle=True, num_workers=1, collate_fn=test_data_collate)

    # for debug use
    # while True:
    #     sample = trainset.__getitem__(np.random.choice(range(trainset.__len__())))
    #     cv2.imshow(sample['text'], inv_transform(sample['image']))
    #     key = cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     if key == 27:
    #         break

    net = crnn(hid_dim=hparams.hidden_dim, chardict=trainset.chardict)
    net.train()
    if cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = hparams.base_lr, weight_decay=0.0001)
    loss_function = CTCLoss()

    ## restore from checkpoint
    if args.restore_from_checkpoint != '':
        ckpt = torch.load(args.restore_from_checkpoint, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt['model_params'])
        optimizer.load_state_dict(ckpt['optim_params'])
        epoch = ckpt['epoch']
    else:
        epoch = 0


    while epoch < hparams.NUM_epochs:
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
            loss = loss_function(preds, labels, pred_lens, label_lens) / hparams.batch_size
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 10.0)

            mean_loss.append(loss.data[0])
            optimizer.step()

            ## set description
            description = 'epoch:{}, iteration:{}, current loss:{}, mean loss:{}'.format(epoch, iter, loss.data[0], np.mean(mean_loss))
            iterator.set_description(description)


        epoch += 1
        if epoch % hparams.snapshot_interval == 0 or epoch == hparams.NUM_epochs:
            ckpt_path = join(ckpt_dir, 'crnn_ckpt_epoch{}'.format(epoch))
            torch.save( {'epoch': epoch, 'loss': loss.data[0],
                         'model_params': net.state_dict(),
                         'optim_params': optimizer.state_dict() },
                        ckpt_path
                        )

        if epoch % hparams.eval_interval == 0 or epoch == hparams.NUM_epochs:
            net.eval()
            count = 0
            avg_editdist = 0
            for test_batch in testloader:
                #test_batch = next(iter(testloader))
                test_batch_size = test_batch['img'].size(0)
                with torch.no_grad():
                    imgs = test_batch['img']
                    if cuda:
                        imgs = imgs.cuda()

                    preds = net(imgs).cpu()
                preds = torch.argmax(preds, dim=2)
                preds = preds.permute(1, 0)

                for i in range(test_batch_size):
                    pred_label = net.seq_to_text(preds[i].tolist())
                    true_label = test_batch['text'][i]
                    avg_editdist += distance(true_label, pred_label)
                    if count == 0:
                        print('true: {}, pred: {}'.format(true_label, pred_label))
                count += 1
            avg_editdist = float(avg_editdist) / testset.__len__()
            print('epoch: {}, average edit distance: {}'.format(epoch, avg_editdist))
            net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='currently only support coco or synthesized')
    parser.add_argument('--restore_from_checkpoint', type=str, default='checkpoints/coco/crnn_ckpt_epoch60', help='path to load checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='gpu info')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    ckpt_dir = os.path.join('checkpoints', args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cuda = True if args.gpu is not '' else False

    train(hparams, args)


