import argparse
from hparams import create_hparams
import os
from model import crnn
from dataset import *
from torchvision.transforms import Compose
import cv2

def test_with_builtin_data(args, hparams):
    transform = Compose([resize(size=(hparams.resize_width, hparams.resize_height)),
                         # gaussian_noise(gauss_mean, gauss_std),
                         ToTensor(),
                         Normalize(hparams.normalize_mean, hparams.normalize_std)
                         ])

    if args.dataset not in ['synthesized', 'coco']:
        raise ValueError('Dataset not supported.')

    if args.dataset == 'coco':
        testset = coco_test(root_dir='cropped_COCO', annotation='desc.json', transform=transform)
    if args.dataset == 'synthesized':
        testset = synthetic_train(height=hparams.syn_height, width=hparams.syn_width,
                                  num_instances=hparams.syn_num_test, transform=transform)

    net = crnn(hid_dim=hparams.hidden_dim, chardict=testset.chardict)
    if cuda:
        net = net.cuda()

    ## restore from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    net.load_state_dict(ckpt['model_params'])
    net.eval()

    for idx in range(testset.__len__()):
        sample = testset.__getitem__(idx)
        img, true_label = sample['image'], sample['text']
        batch_img = torch.unsqueeze(img, 0)
        if cuda:
            batch_img = batch_img.cuda()
        pred = net(batch_img).cpu()
        pred = torch.argmax(pred, dim=2)
        pred = pred.permute(1, 0)

        pred_label = net.seq_to_text(pred[0].tolist())

        cv2.imshow("true:{}, pred:{}".format(true_label, pred_label), inv_transform(img))
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:
            break

    print('exiting..')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='currently only support coco or synthesized')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/coco/crnn_ckpt_epoch60',
                        help='path to load checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='gpu info')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    parser.add_argument('--mode', type=str, default='live')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    ckpt_dir = os.path.join('checkpoints', args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cuda = True if args.gpu is not '' else False

    test_with_builtin_data(args, hparams)









