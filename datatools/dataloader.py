from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader
from datatools.dataset import PatchData, MaskData
import my_transforms
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1):
        for t in self.transforms:
            img1 = t(img1)
        return img1

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class MinibatchSampler(data.Sampler):
    def __init__(self, dataset, SEED=2333):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)

def getmask_training_dataloader(img1, img2, mask, seed, num_size, samplerstate=False):
    transform_train = Compose([
        transforms.RandomChoice([
            my_transforms.Noprocess(),

            ]),

    ])

    train_data = MaskData(img1, img2, mask, seed, num_size, transform=transform_train, Training=True)
    sampler = None
    if samplerstate:
        sampler = MinibatchSampler(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, sampler=sampler, batch_sampler=None, num_workers=0)
    return train_loader

def getmask_test_dataloader(img1, img2, mask, samplerstate=False):
    transform_test = Compose([
        transforms.RandomChoice([
                my_transforms.Noprocess(),
            ]),
    ])
    test_data = MaskData(img1, img2, mask, transform=transform_test, Training=False)
    sampler = None
    if samplerstate:
        sampler = MinibatchSampler(test_data)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, sampler=sampler, batch_sampler=None, num_workers=0)
    return test_loader




def getmask_all_dataloader(img1, img2, mask):
    transform_all = None
    all_data = MaskData(img1, img2, mask, transform=transform_all, Training=False)
    all_data_loader = DataLoader(dataset=all_data, batch_size=1, shuffle=False, batch_sampler=None, num_workers=0)
    return all_data_loader



