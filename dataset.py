import torch
from torch.utils.data import Dataset
import json
from os.path import join
import cv2


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

class coco_test(Dataset):
    def __init__(self, root_dir, annotation, transform=None):
        with open(join(root_dir, annotation)) as f:
            myanno = json.load(f)

        self.mydict = myanno['abc']  # type: str
        self.root_dir = root_dir
        self.annotation = myanno['test']  # list of dicts {'text':..., 'name':...}
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_name = self.annotation[idx]['name']
        text = self.annotation[idx]['text']

        img = cv2.imread(join(self.root_dir, img_name))  # shape: row * col * channels

        #seq = self.text_to_seq(text)
        #sample = {'text': seq, 'image': img}
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


# strategy to combine a batch of training data
def train_data_collate(batch):
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

# strategy to combine a batch of test data
def test_data_collate(batch):
    img = []
    seq = []

    for sample in batch:
        img.append(torch.from_numpy(sample['image'].transpose((2, 0, 1))).float())
        seq.append(sample['text'])

    img = torch.stack(img)

    batch = {"img": img, "seq": seq}
    return batch