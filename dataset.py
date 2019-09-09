import torch
from torch.utils.data import Dataset
import json
from os.path import join
import cv2
import numpy as np
import string
import torchvision.transforms


class synthetic_train(Dataset):
    def __init__(self, height, width, chardict=string.ascii_letters+string.digits ,num_instances=20000, min_numchars=1, max_numchars=10, transform=None):
        self.size = (height, width, 3)
        self.chardict = chardict
        self.num_instances = num_instances
        self.min_numchars = min_numchars
        self.max_numchars = max_numchars
        self.transform = transform

        self.texts = []
        self.images = []
        for i in range(self.num_instances):
            img, text = self.create_image_with_text()
            self.texts.append(text)
            self.images.append(img)

        return

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        img = self.images[idx]
        text = self.texts[idx]
        seq = self.text_to_seq(text)
        sample = {'image':img, 'text':text, 'seq':seq}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def create_image_with_text(self):
        img = np.zeros( self.size, dtype=np.uint8 )

        text_leng = np.random.choice(range(self.min_numchars,self.max_numchars+1,1))
        text_idx = np.random.choice(range(len(self.chardict)), text_leng)
        text = ''
        for i in text_idx:
            text = text + self.chardict[i]
        w, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

        bottom_left_x = np.random.choice(range(1,max(self.size[1]-w,2),1))
        bottom_left_y = np.random.choice(range(h,max(self.size[0],h+1),1))

        cv2.putText(img, text, (bottom_left_x,bottom_left_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), False)
        return img, text

    def text_to_seq(self, text):
        return [self.chardict.find(char)+1 for char in text if self.chardict.find(char)>=0]   # 0 means non-character


class coco_train(Dataset):
    def __init__(self, root_dir, annotation, transform=None):
        with open(join(root_dir, annotation)) as f:
            myanno = json.load(f)

        self.chardict = myanno['abc']  # type: str
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
        sample = {'text': text, 'image': img, 'seq':seq}
        if self.transform:
            sample = self.transform(sample) # shape: predefined_height * predefined_width * channels

        return sample

    def text_to_seq(self, text):
        return [self.chardict.find(char)+1 for char in text if self.chardict.find(char)>=0]   # 0 means non-character

class coco_test(Dataset):
    def __init__(self, root_dir, annotation, transform=None):
        with open(join(root_dir, annotation)) as f:
            myanno = json.load(f)

        self.chardict = myanno['abc']  # type: str
        self.root_dir = root_dir
        self.annotation = myanno['test']  # list of dicts {'text':..., 'name':...}
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_name = self.annotation[idx]['name']
        text = self.annotation[idx]['text']
        img = cv2.imread(join(self.root_dir, img_name))  # shape: row * col * channels
        sample = {'text': text, 'image': img}

        if self.transform:
            sample = self.transform(sample)  # shape: predefined_height * predefined_width * channels

        return sample


# data transform
class resize:
    def __init__(self, size=(224, 32)):
        self.size = size

    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.size)
        return sample

class gaussian_noise:
    def __init__(self, mu=20.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        img = sample['image']

        noise = np.random.randn(*img.shape)
        noise = self.sigma * noise + self.mu

        noisy_img = img.astype(np.float)
        noisy_img[img < 240] += noise[img < 240]
        sample['image'] = noisy_img.astype(np.uint8)

        return sample

class ToTensor:
    def __init__(self):
        pass
    def __call__(self, sample):
        totensor = torchvision.transforms.ToTensor()
        sample['image'] = totensor(sample['image'])
        #sample['image'] = torch.from_numpy(sample['image'].transpose((2,0,1))).float()
        return sample

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        sample['image'] = normalize(sample['image'])
        return sample



# strategy to combine a batch of training data
def train_data_collate(batch):
    img = []
    seq = []
    seq_len = []
    text = []
    for sample in batch:
        img.append(sample['image'])
        seq.extend(sample['seq'])
        seq_len.append(len(sample['seq']))
        text.append(sample['text'])

    img = torch.stack(img)
    seq = torch.Tensor(seq).int()
    seq_len = torch.Tensor(seq_len).int()

    batch = {"img": img, "seq": seq, "seq_len": seq_len, 'text': text}
    return batch

# strategy to combine a batch of test data
def test_data_collate(batch):
    img = []
    text = []

    for sample in batch:
        img.append(sample['image'])
        text.append(sample['text'])

    img = torch.stack(img)
    batch = {"img": img, "text": text}
    return batch


def inv_transform(img_tensor):
    img = img_tensor.numpy()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]

    img = img * 255.0
    img = img.transpose((1,2,0))
    return img.astype(np.uint8)



