import pickle
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
# import pandas as pd
import numpy as np
from read_xray_once import read_image


def divide_dataset(filenames):
    """
    The data will be divided into trainset and testset
    :param filenames: x1_name & y1: all data selected from original 2 dataset
                      x2, y2: x_generated & y_generated with label 1
    :return: [x_train, y_train], [x_test, y_test]
    """
    x1, x2, y1, y2 = filenames
    x1_names = np.load(x1)
    x2 = np.load(x2)
    y1 = np.load(y1)
    y2 = np.load(y2)
    x, y = [], []
    for i in range(0, y1.shape[0]):
        if y1[i][0] != 1:  # x2 has all data with label 1
            image = read_image(x1_names[i][0], is_covid=False)
            x = x + image
            y.append(y1[i][0])
    x = np.concatenate((np.array(x), x2), axis=0)
    y = np.concatenate((np.expand_dims(np.array(y), axis=1), y2), axis=0)

    trainset, testset = [None, None], [None, None]
    trainset[0], testset[0], trainset[1], testset[1] = train_test_split(x, y,
                                                                        test_size=0.2,
                                                                        random_state=2021,
                                                                        stratify=y)
    return trainset, testset


class Xray_dataset(Dataset):
    def __init__(self, data, trans=None, is_train=True, ):
        """
        :param data: dataset
        :param trans: specific transform
        :param is_train: trainset or testset
        """
        self.data = data
        self.is_train = is_train

        if trans is None:
            if self.is_train:
                mean_std = [np.mean(self.data[0]), np.std(self.data[0])]
                normalize = T.Normalize(mean=[mean_std[0] / 255],
                                        std=[mean_std[1] / 255])

                with open('./original_data/X-Ray/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'
                          '/mean_std255.txt', 'wb') as f:
                    pickle.dump(mean_std, f)
                self.transforms = T.Compose([
                    T.ToPILImage(),
                    # T.Resize(224),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                    normalize
                ])
            else:
                with open('./original_data/X-Ray/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'
                          '/mean_std255.txt', 'rb') as f:
                    mean_std = pickle.load(f)
                normalize = T.Normalize(mean=[mean_std[0] / 255],
                                        std=[mean_std[1] / 255])
                self.transforms = T.Compose([
                    T.ToPILImage(),
                    # T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = trans

    def __getitem__(self, index):
        """

        :param index:
        :return: data and label
        """
        x = self.transforms(self.data[0][index])
        # x = np.expand_dims(x, 0)
        return x, self.data[1][index][0]

    def __len__(self):
        """

        :return: length of dataset
        """
        return self.data[1].shape[0]


prefix = './original_data/X-Ray/'
data_folder = prefix + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
filename = [data_folder+'x_filenames.npy', data_folder+'x_generated.npy',
            data_folder+'y_original.npy', data_folder+'y_generated.npy', ]

if __name__ == '__main__':
    trainset, testset = divide_dataset(filename)
    np.save('./dataset/xray/x_train.npy', trainset[0])
    np.save('./dataset/xray/y_train.npy', trainset[1])
    np.save('./dataset/xray/x_test.npy', testset[0])
    np.save('./dataset/xray/y_test.npy', testset[1])

    print(80)
