import os, cv2, torch
from skimage import io
import numpy as np
import random
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils import divisible_pad
import matplotlib.pyplot as plt
import spectral
from PIL import Image
from my_transforms import flip, rotate, radiation_noise
# from random import shuffle



def minibatch_sample(gt_mask: np.ndarray, batch_size, seed):
    rs = np.random.RandomState(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)



    inds_dict_per_class = dict()
    for cls in range(len(cls_list)-1):
        inds = np.argwhere(gt_mask == cls+1)
        np.random.seed(seed)
        np.random.shuffle(inds)
        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(gt_mask)
        for cls, inds in inds_dict_per_class.items():
            np.random.seed(seed)
            np.random.shuffle(inds)
            cd = min(batch_size, len(inds))
            fetch_inds = inds[:cd]
            train_inds[fetch_inds] = cls+1

        cnt += 1
        if cnt == 11:
            return train_inds_list

        train_inds_list.append(train_inds.reshape(gt_mask.shape))



class MaskData(Dataset):
    def __init__(self, data1, data2, mask, np_seed=2333, batch_size=10, transform=None, Training=False):
        self.train_data1 = data1
        self.train_data2 = data2

        self.training = Training
        self.gt_mask = mask
        self.batch_size = batch_size
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.transform = transform
        self.augmentation = True


    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.gt_mask, self.batch_size, seed=self.seeds_for_minibatchsample.pop())


    def __getitem__(self, index):
        image_1 = self.train_data1
        image_2 = self.train_data2
        gt_mask = self.gt_mask

        gt_mask = self.gt_mask[np.newaxis, :, :]
        if self.augmentation and self.training:
            if np.random.random() > 0.5:
                image_1, image_2, gt_mask = flip(image_1, image_2, gt_mask)
            if np.random.random() > 0.5:
                image_1, image_2, gt_mask = rotate(image_1, image_2, gt_mask)
        gt_mask = np.squeeze(gt_mask)

        image_1 = torch.from_numpy(np.copy(image_1)).type(torch.FloatTensor)
        image_2 = torch.from_numpy(np.copy(image_2)).type(torch.FloatTensor)
        gt_mask = torch.from_numpy(np.copy(gt_mask)).type(torch.LongTensor)

        return image_1.cuda(), image_2.cuda(), gt_mask.cuda()-1

    def __len__(self):

        return np.argwhere(self.gt_mask != 0).shape[1]



def cumulativehistogram(array_data, counts, percent):
    """累计直方图统计""" # 逐波段统计最值
    gray_level, gray_num = np.unique(array_data, return_counts=True)
    count_percent1 = counts * percent
    count_percent2 = counts * (1 - percent)
    cutmax = 0
    cutmin = 0

    for i in range(1, len(gray_level)):
        gray_num[i] += gray_num[i - 1]
        if (gray_num[i] >= count_percent1 and gray_num[i - 1] <= count_percent1):
            cutmin = gray_level[i]
        if (gray_num[i] >= count_percent2 and gray_num[i - 1] <= count_percent2):
            cutmax = gray_level[i]
    return cutmin, cutmax


def preprocess(image, percent):
    """数值截断，进行归一化"""
    c, h, w = image.shape[0], image.shape[1], image.shape[2]
    array_data = image[:, :, :]

    compress_data = np.zeros((c, h, w))
    for i in range(c):

        cutmin, cutmax = cumulativehistogram(array_data[i, :, :], h*w, percent)
        compress_scale = cutmax - cutmin
        if compress_scale == 0:
            print('error')

        temp = np.array(array_data[i, :, :])
        temp[temp > cutmax] = cutmax
        temp[temp < cutmin] = cutmin
        compress_data[i, :, :] = (temp - cutmin) / (cutmax - cutmin)
    return compress_data


def padimg(image, size):
    """图片边缘镜像填充"""
    Interpolation = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REPLICATE： 进行复制的补零操作;
    # cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
    # cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
    # cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdegh|abcdefgh|abcdefg;
    top_size, bottom_size, left_size, right_size = (size, size,
                                                    size, size)
    image = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, Interpolation)
    return image



def getdata_sample(data_name, percent, size, mode):
    """读取图片、标签
    tif格式图片读取
    data1_tif = TIFF.open('./data/Augsburg/MS_fusion.tif', mode='r')
    data1_np = data1_tif.read_image()
    print('原始data1的形状：', np.shape(data1_np))
    """
    print('-------------------数据读取中-------------------')
    # mat格式图片读取
    data_dir = '../dataset/'+ data_name + '/'
    if data_name == 'MUUFL':
        data1 = loadmat(os.path.join(data_dir, 'data_HS_LR.mat'))['hsi_data']
        data2 = loadmat(os.path.join(data_dir, 'data_SAR_HR.mat'))['lidar_data']
        labels = loadmat(os.path.join(data_dir, 'gt.mat'))['labels']
        where_0 = np.where(labels == -1)
        labels[where_0] = 0
    elif data_name == '2013':
        data1 = loadmat(os.path.join(data_dir, 'HSI.mat'))['HSI']
        data2 = loadmat(os.path.join(data_dir, 'LiDAR.mat'))['LiDAR']
        labels = loadmat(os.path.join(data_dir, 'gt.mat'))['gt']
    else:
        data1 = loadmat(os.path.join(data_dir, 'data_HS_LR.mat'))['data_HS_LR']
        data2 = loadmat(os.path.join(data_dir, 'data_SAR_HR.mat'))['data_SAR_HR']
        labels = loadmat(os.path.join(data_dir, 'gt.mat'))['gt']


    # 图片数据类型转换
    if data1.dtype=='float32' or data1.dtype=='float64':
        data1 = data1 * 10000
    if data2.dtype=='float32' or data2.dtype=='float64':
        data2 = data2 * 100

    data1 = data1.astype(int)
    data2 = data2.astype(int)
    labels = labels.astype(np.uint8)
    print('已读取')



    print('填充前：', (np.shape(data1), np.shape(data2)))
    if len(np.shape(data1)) < 3:
        data1 = np.expand_dims(data1, axis=2)
    if len(np.shape(data2)) < 3:
        data2 = np.expand_dims(data2, axis=2)


    data1 = np.array(data1).transpose((2, 0, 1))
    data2 = np.array(data2).transpose((2, 0, 1))
    data11 = preprocess(data1, percent)
    data22 = preprocess(data2, percent)



    blob = divisible_pad([np.concatenate([data11,
                                          data22,
                                          labels[None, :, :]], axis=0)], 16, False)

    data1 = blob[0, :data11.shape[0], :, :]
    data2 = blob[0, data11.shape[0]:data11.shape[0]+data22.shape[0], :, :]
    labels = blob[0, -1, :, :]



    print('已填充')
    print('填充后：', (np.shape(data1), np.shape(data2)))

    if len(np.shape(data1)) < 3:
        data1 = np.expand_dims(data1, axis=2)
    if len(np.shape(data2)) < 3:
        data2 = np.expand_dims(data2, axis=2)

    print('已归一化')

    return data1, data2, labels


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

def getdata_fixed(data_name, percent, size, mode):
    """读取图片、标签
       tif格式图片读取
       data1_tif = TIFF.open('./data/Augsburg/MS_fusion.tif', mode='r')
       data1_np = data1_tif.read_image()
       print('原始data1的形状：', np.shape(data1_np))
       """
    print('-------------------数据读取中-------------------')
    data_dir = '../dataset/' + data_name + '/'

    if data_name == '2013':
        data1 = loadmat(os.path.join(data_dir, 'HSI.mat'))['HSI']
        data2 = loadmat(os.path.join(data_dir, 'LiDAR.mat'))['LiDAR']
        labels_train = loadmat(os.path.join(data_dir, 'TRLabel.mat'))['TRLabel']
        labels_test = loadmat(os.path.join(data_dir, 'TSLabel.mat'))['TSLabel']
    else:
        data1 = loadmat(os.path.join(data_dir, 'data_HS_LR.mat'))['data_HS_LR']
        data2 = loadmat(os.path.join(data_dir, 'data_SAR_HR.mat'))['data_SAR_HR']
        labels_train = loadmat(os.path.join(data_dir, 'TrainImage.mat'))['TrainImage']
        labels_test = loadmat(os.path.join(data_dir, 'TestImage.mat'))['TestImage']



    # 图片数据类型转换
    if data1.dtype == 'float32' or data1.dtype == 'float64':
        data1 = data1 * 100
    if data2.dtype == 'float32' or data2.dtype == 'float64':
        data2 = data2 * 100

    data1 = data1.astype(int)
    data2 = data2.astype(int)
    labels_train = labels_train.astype(np.uint8)
    labels_test = labels_test.astype(np.uint8)
    print('已读取')


    print('填充前：', (np.shape(data1), np.shape(data2)))
    if len(np.shape(data1)) < 3:
        data1 = np.expand_dims(data1, axis=2)
    if len(np.shape(data2)) < 3:
        data2 = np.expand_dims(data2, axis=2)



    data1 = np.array(data1).transpose((2, 0, 1))
    data2 = np.array(data2).transpose((2, 0, 1))

    data11 = preprocess(data1, percent)
    data22 = preprocess(data2, percent)

    blob = divisible_pad([np.concatenate([data11,
                                          data22,
                                          labels_train[None, :, :],
                                          labels_test[None, :, :]
                                          ], axis=0)], 16, False)

    data1 = blob[0, :data11.shape[0], :, :]
    data2 = blob[0, data11.shape[0]:data11.shape[0]+data22.shape[0], :, :]
    labels_train = blob[0, -2, :, :]
    labels_test = blob[0, -1, :, :]



    print('已填充')
    print('填充后：', (np.shape(data1), np.shape(data2)))

    if len(np.shape(data1)) < 3:
        data1 = np.expand_dims(data1, axis=2)
    if len(np.shape(data2)) < 3:
        data2 = np.expand_dims(data2, axis=2)


    print('已归一化')

    return data1, data2, labels_train, labels_test


def splitmaskdata_sample(labels, Traindata_Rate, Traindata_Num, Split_MODE, SEED):

    label_list, labels_counts = np.unique(labels, return_counts=True)  # 返回类别标签与各个类别所占的数量
    all_labeled_num = len(np.argwhere(labels != 0))


    print('-------------------数据划分----------------------')
    print('类标：', label_list)
    print('各类样本数：', labels_counts)
    Cls_Number = len(label_list) - 1
    print('标注的类别数：', Cls_Number)
    ground_xy = np.array([[]] * Cls_Number).tolist()


    if Split_MODE == 'Cls_Rate_Same':
        split_num_list = np.ceil(labels_counts * Traindata_Rate).astype(np.int)
    if Split_MODE == 'Cls_Num_Same':
        split_num_list = [Traindata_Num] * (Cls_Number + 1)

    print(split_num_list)
    index_train_data = []
    index_test_data = []

    train_indicator = np.zeros_like(labels)
    test_indicator = np.zeros_like(labels)

    for id in range(Cls_Number):
        label = id + 1
        ground_xy[id] = np.argwhere(labels == label)

        np.random.seed(SEED)
        np.random.shuffle(ground_xy[id])
        categories_number = labels_counts[label]
        split_num = split_num_list[label]

        index_train_data.extend(ground_xy[id][:split_num])
        index_test_data.extend(ground_xy[id][split_num:])

    train_indicator[tuple(zip(*index_train_data))] = 1
    test_indicator[tuple(zip(*index_test_data))] = 1



    label_train = train_indicator * labels
    label_test = test_indicator * labels

    label_list, labels_counts = np.unique(label_train, return_counts=True)
    print(label_list)
    print(labels_counts)

    label_list, labels_counts = np.unique(label_test, return_counts=True)
    print(label_list)
    print(labels_counts)


    return label_train, label_test
