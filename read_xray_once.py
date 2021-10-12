import pandas as pd
import cv2
import numpy as np
import random
import os


def read_image(file, is_covid=False, size=256):  # cv2 takes up too much memory
    """
    read the original file of Chest-XRay
    :param file: path of file
    :param is_covid: it doesnot make sense after modification
    :param size: crop the image
    :return: [one 2D image]
    """
    if not is_covid:
        # image = cv2.imread(file)  # (H, W, C)
        image = cv2.imread(file, 0)  # (H, W)
        # cv2.namedWindow('im')
        # cv2.imshow('im', image)
        # cv2.waitKey(27)
        # cv2.destroyWindow('im')
        image = image[int(0.1 * image.shape[0]):int(0.9 * image.shape[0]),  # crop
                int(0.1 * image.shape[1]):int(0.9 * image.shape[1])]
        image = cv2.resize(image, (size, size))
        # return np.transpose(image, (2, 0, 1))  # (C, H, W) for subsequent tensor
        return [image]
    else:
        image = cv2.imread(file)  # (H, W, C)
        image = np.transpose(image, (2, 0, 1))
        temp = [None, None, None]
        for i in range(0, 3):
            temp[i] = image[i]
            temp[i] = temp[i][int(0.1 * temp[i].shape[0]):int(0.9 * temp[i].shape[0]),  # crop
                    int(0.1 * temp[i].shape[1]):int(0.9 * temp[i].shape[1])]
            temp[i] = cv2.resize(temp[i], (size, size))
        '''
        diff = []  # prove they are same actually
        for i in range(0, 256):
            for j in range(0, 256):
                
                if temp[0][i][j] != temp[1][i][j]:
                    diff.append((i, j))
        '''
        return [temp[0]]


# https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_Metadata.csv
def read_xray_filename(is_train):
    """
    read filename of Coronahack-Chest-XRay-Dataset based on metadata
    :param is_train: trainset or testset
    :return: x & y of Coronahack-Chest-XRay-Dataset
    """
    if is_train:
        selected = 'TRAIN'
    else:
        selected = 'TEST'
    prefix = './original_data/X-Ray/'
    meta = prefix + 'Chest_xray_Corona_Metadata.csv'
    data_folder = prefix + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
    meta_df = pd.read_csv(meta)

    x = []
    y = []
    for i in range(0, meta_df.shape[0]):
        temp = meta_df.iloc[i]
        if temp[3].upper() != selected:
            continue

        label = temp[2].lower()
        if label == 'normal':
            y.append(0)
        elif label == 'pnemonia':
            if temp[4] == 'COVID-19':
                y.append(1)
            else:
                y.append(2)

        '''
        # path = data_folder + 'train/' if temp[3].upper() == 'TRAIN' else data_folder + 'test/'
        path = data_folder + selected.lower() + '/'
        # print(29)
        # x.append(read_image(path + temp[1]))  # (C, H, W) for subsequent tensor
        if temp[4] == 'COVID-19':  # generate three images
            x = x + read_image(path + temp[1], is_covid=True)
            y.append(1)
            y.append(1)
        else:
            x = x + read_image(path + temp[1])
    print(12)
    return np.array(x), np.array(y)
    '''
        path = data_folder + selected.lower() + '/'
        x.append(path + temp[1])
    # counts = {}
    # for i in y:
    #     counts[i] = counts.get(i, 0) + 1
    return x, y


class Augmentation(object):
    def __init__(self, images, new_num=2):
        """
        generate new images based on input images
        :param images: original images input
        :param new_num: number of images to be generated based on each input
        """
        self.images = images
        self.increasing_num = new_num
        self.x, self.y = [], []

        # self.generate()

    # https://zhuanlan.zhihu.com/p/45413962
    # https://blog.csdn.net/qq_37674858/article/details/80708393
    # https://www.cxyzjd.com/article/qq_45769063/107137025
    def generate(self):
        for image in self.images:
            height, width = image.shape
            for i in range(0, self.increasing_num):
                if random.random() > 0.5:
                    image = np.fliplr(image)  # flip the image
                if random.random() > 0.5:  # horizontal translation
                    image = self.horizontal_translation(image, height, width)
                # if random.random() > 0.5:  # vertical translation
                #     image = self.vertical_translation(image, height, width)
                if random.random() > 0.6:  # center rotation
                    image = self.rotation(image, height, width, )
                if random.random() > 0.7:  # add gauss noise
                    image = self.gauss_noise(image, )
                self.x.append(image)
                self.y.append(1)

        return self.x, self.y

    def horizontal_translation(self, image, height, width):
        distance = random.randint(int(0.05*width), int(0.1*width))
        if random.random() > 0.5:  # Left
            for i in range(0, height):
                for j in range(0, width-distance):
                    image[i][j] = image[i][j + distance]
                for j in range(width-distance, width):
                    image[i][j] = 0
        else:  # right
            for i in range(0, height):
                for j in range(width-1, distance, -1):
                    image[i][j] = image[i][j - distance]
                for j in range(distance, 0, -1):
                    image[i][j] = 0
        return image

    def vertical_translation(self, image, height, width):
        distance = random.randint(int(0.05*height), int(0.1*height))
        if random.random() > 0.5:  # up
            for j in range(0, width):
                for i in range(0, height-distance):
                    image[i][j] = image[i + distance][j]
                for i in range(height-distance, height):
                    image[i][j] = 0
        else:  # down
            for j in range(0, width):
                for i in range(height-1, distance, -1):
                    image[i][j] = image[i - distance][j]
                for i in range(distance, 0, -1):
                    image[i][j] = 0
        return image

    def rotation(self, image, height, width, angle=None):
        center = (width // 2, height // 2)
        if angle is None:
            angle = random.randint(5, 15)
        if random.random() > 0.5:  # clockwise
            rotation_matrix = cv2.getRotationMatrix2D(center,
                                                      -angle, scale=1.0)
        else:  # Counterclockwise
            rotation_matrix = cv2.getRotationMatrix2D(center,
                                                      angle, scale=1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return image

    def gauss_noise(self, image, mean=0, var=0.0001):
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)  # limit the values of the new image
        out = np.uint8(out * 255)
        return out


def read_x2():
    """
    read another X Ray data
    :return: x's filename & y
    """
    prefix = './original_data/Data/'
    train_path = prefix + 'train'
    test_path = prefix + 'test'
    x = []
    y = []
    for dirpath, dirnames, filenames in os.walk(prefix):
        if filenames:
            if train_path in dirpath:
                label = dirpath[len(train_path)+len('\\'):]
            elif test_path in dirpath:
                label = dirpath[len(test_path)+len('\\'):]
            if label == 'COVID19':
                for i in range(0, len(filenames)):
                    x.append(dirpath + '/' + filenames[i])
                    y.append(1)
            elif label == 'NORMAL':
                for i in range(0, len(filenames)):
                    x.append(dirpath + '/' + filenames[i])
                    y.append(0)
            elif label == 'PNEUMONIA':
                for i in range(0, len(filenames)):
                    x.append(dirpath + '/' + filenames[i])
                    y.append(2)
        # print(180)
    return x, y


def filter_covid(x_filename, y):
    """
    filter different categories of dataset
    :param x_filename: filename/path of x
    :param y: label
    :return: x with covid, x0's filename, x1's filename, x2's filename
    """
    index0 = []
    index1 = []
    index2 = []
    for i in range(0, len(y)):
        if y[i] == 0:
            index0.append(i)
        elif y[i] == 1:
            index1.append(i)
        elif y[i] == 2:
            index2.append(i)
    x_covid = []
    for index in index1:
        x_covid = x_covid + read_image(x_filename[index], is_covid=True)
    return x_covid, index0, index1, index2


def select_data(x_filename, y, index0, index1, index2):
    """
    select a specific amount of data
    :param x_filename: filename/path of x
    :param y: label
    :param index0: index of data with label=0
    :param index1: index of data with label=1
    :param index2: index of data with label=2
    :return: x's filename & y
    """
    random.shuffle(index0)
    random.shuffle(index1)
    random.shuffle(index2)
    index = []
    index += index0[:1000] + index1 + index2[:1000]
    x_new, y_new = [], []
    for i in index:
        x_new.append(x_filename[i])
        y_new.append(y[i])
    return x_new, y_new


if __name__ == '__main__':
    x_train_xray, y_train_xray = read_xray_filename(is_train=True)  # 5286:1342 58 3886
    x_test_xray, y_test_xray = read_xray_filename(is_train=False)  # 624:234 0 390
    # There is no COVID-19 sample in test set, so it's necessary to shuffle them
    x_filename = np.concatenate((x_train_xray, x_test_xray), axis=0)  # 5910
    y = np.concatenate((y_train_xray, y_test_xray), axis=0)

    x2_filename, y2 = read_x2()  # 6432

    x_covid, x_index0, x_index1, x_index2 = filter_covid(x_filename, y)  # 1576 58 4276
    x2_covid, x2_index0, x2_index1, x2_index2 = filter_covid(x2_filename, y2)  # 1583 576 4273
    x_covid += x2_covid  # 634

    x_new, y_new = select_data(x_filename, y, x_index0, x_index1, x_index2)  # 2058
    x2_new, y2_new = select_data(x2_filename, y2, x2_index0, x2_index1, x2_index2)  # 2576

    x_generated, y_generated = Augmentation(x_covid).generate()
    x_generated, y_generated = np.array(x_generated), np.array(y_generated)

    prefix = './original_data/X-Ray/'
    data_folder = prefix + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
    # x_filename = np.expand_dims(x_filename, axis=1)
    # x_generated.tofile(data_folder+'x_generated.csv', sep='', format='%10.5f')
    # pd.DataFrame(np.expand_dims(y_generated, axis=1)).to_csv(data_folder + 'y_generated.csv')
    # pd.DataFrame(np.expand_dims(x_filename, axis=1)).to_csv(data_folder + 'x_filenames.csv')
    # pd.DataFrame(np.expand_dims(y, axis=1)).to_csv(data_folder + 'y_original.csv')
    np.save('./dataset/xray/' + 'x_generated.npy', x_generated)
    np.save('./dataset/xray/' + 'y_generated.npy', np.expand_dims(y_generated, axis=1))
    np.save('./dataset/xray/' + 'x_filenames.npy', np.expand_dims(x_new+x2_new, axis=1))
    np.save('./dataset/xray/' + 'y_original.npy', np.expand_dims(y_new+y2_new, axis=1))

    print(79)
