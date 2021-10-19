import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from sklearn.decomposition import PCA


class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(Residual_block, self).__init__()
        self.essential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = shortcut

    def forward(self, x):
        out = self.essential(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


def make_layer(in_channels, out_channels, num_block, stride=1):
    """

    :param in_channels:
    :param out_channels:
    :param num_block:
    :param stride: the stride of the first residual block in a layer.
    :return :
    """
    layers = []
    if stride != 1:
        # dotted line skip, indicating the increase of dimension/channel
        shortcut = nn.Sequential(
            # k=1,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
    else:
        # solid line skip
        shortcut = None
    layers.append(Residual_block(in_channels, out_channels, stride, shortcut))
    # solid line skip
    for i in range(1, num_block):
        layers.append(Residual_block(out_channels, out_channels))

    return nn.Sequential(*layers)


class Resnet(nn.Module):
    """
    the input tensor data should be [batch_size, channel, 224, 224].
    """
    def __init__(self, in_channel, num_labels=3):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            # (224+2p-k)//2 + 1 = c, k=7, c=112, so p=3
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # process the first maxpool separately, then it's convenient to divide the residual block
            # (112+2p-k)//2 + 1 = c, k=3, c=56, so p=1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 56*56*64 -> 56*56*64, s=1
        self.layer1 = make_layer(64, 64, num_block=3)
        # 56*56*64 -> 28*28*128, s=2
        self.layer2 = make_layer(64, 128, num_block=4, stride=2)
        # 28*28*128 -> 14*14*256, s=2
        self.layer3 = make_layer(128, 256, num_block=6, stride=2)
        # 14*14*256 -> 7*7*512, s=2
        self.layer4 = make_layer(256, 512, num_block=3, stride=2)

        # dense
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        xx = self.pre(x)

        xx = self.layer1(xx)
        xx = self.layer2(xx)
        xx = self.layer3(xx)
        xx = self.layer4(xx)

        # pool = nn.AvgPool2d(kernel_size=7)
        # xx = pool(xx)
        xx = F.avg_pool2d(xx, kernel_size=7)

        # print(xx.shape)
        xx = xx.view(xx.shape[0], -1)
        # print(xx.shape)

        return self.fc(xx)


class Pretrained_res50(nn.Module):
    def __init__(self, device='cpu', para=None, ):
        super(Pretrained_res50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # modify the input
        # print(resnet50.conv1) Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        pretrained_dict = resnet50.state_dict()  # 'conv1.weight'
        weights = pretrained_dict['conv1.weight']
        weights.size()
        gray = torch.zeros(64, 1, 7, 7)
        for i, output_channel in enumerate(weights):
            # Gray = 0.299R + 0.587G + 0.114B
            gray[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_dict['conv1.weight'] = gray
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet50.load_state_dict(pretrained_dict)
        for param in resnet50.parameters():
            param.requires_grad = False
        # modify the last FC
        fc_inputs = resnet50.fc.in_features  # 2048
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
            # nn.LogSoftmax(dim=1)
        )
        if para:
            resnet50.load_state_dict(torch.load(para, map_location=device))
            for param in resnet50.parameters():
                param.requires_grad = False
        self.model = resnet50.to(device)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        return self.model(x)


class Pretrained_vgg(nn.Module):
    def __init__(self, device='cpu', para=None):
        super(Pretrained_vgg, self).__init__()
        vgg = models.vgg.vgg16(pretrained=True)
        # vgg.features[0]  # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # modify the input
        pretrained_dict = vgg.state_dict()  # 'features.0.weight'
        weights = pretrained_dict['features.0.weight']
        gray = torch.zeros(64, 1, 3, 3)
        for i, output_channel in enumerate(weights):
            # Gray = 0.299R + 0.587G + 0.114B
            gray[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_dict['features.0.weight'] = gray
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), )
        vgg.load_state_dict(pretrained_dict)
        for param in vgg.parameters():
            param.requires_grad = False
        # modify the last FC. vgg.classifier[6] Linear(in_features=4096, out_features=1000, bias=True)
        vgg.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )
        if para:
            vgg.load_state_dict(torch.load(para, map_location=device))
            for param in vgg.parameters():
                param.requires_grad = False
        self.model = vgg.to(device)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        return self.model(x)


class Ensemble(nn.Module):
    def __init__(self, paras, device='cpu', dnn_param=None, ):
        """
        build an ensemble model based on ResNet-50 and vgg.
        :param paras: [parameter path of ResNet, of vgg]
        :param device:
        :param dnn_param: parameter path of final DNN
        """
        super(Ensemble, self).__init__()
        self.res = Pretrained_res50(device, paras[0])
        self.res.model.fc = self.res.model.fc[:-2]  # output 256
        self.vgg = Pretrained_vgg(device, paras[1])
        self.vgg.model.classifier[6] = self.vgg.model.classifier[6][:-2]  # output 256
        dnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )
        if dnn_param:
            # dnn.load_state_dict(torch.load('./dataset/xray_dnn_parameters.pkl'))
            dnn.load_state_dict(torch.load(dnn_param, map_location=device))
            for param in dnn.parameters():
                param.requires_grad = False
        self.dnn = dnn.to(device)
        print(198)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        res_out = self.res(x)
        vgg_out = self.vgg(x)
        inputs = torch.cat((res_out, vgg_out), 1)
        return self.dnn(inputs)


class Extract_features(object):
    def __init__(self, batch_size):
        """
        extract HoG features / PCA features.
        :param batch_size: of data to define the PCA
        """
        super(Extract_features, self).__init__()
        '''
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        n_bins = 9
        '''
        win_size = (224, 224)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        n_bins = 8
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
        # !pip install opencv-contrib-python==4.3.0.36
        # pip install opencv-python==3.4.1.15
        # pip install opencv-contrib-python==3.4.1.15
        # sift = cv2.xfeatures2d.SIFT_create(256)
        self.pca = PCA(n_components=batch_size)

    def extract_hog(self, data):
        # batch = data.shape[0]
        features = []
        for i in range(0, data.shape[0]):
            img = data[i][0]
            # print(img.numpy().dtype)
            des = self.hog.compute(img.cpu().numpy().astype(np.uint8), winStride=(16, 16))
            features.append(des)  # des.shape(5408, 1)

        features = np.array(features).reshape((len(features), features[0].shape[0]))
        return features

    def extract_pca(self, data):
        new_features = self.pca.fit_transform(self.extract_hog(data))
        # print(242)
        return new_features


class Ensemble2(nn.Module):
    def __init__(self, paras, dnn1_param, device='cpu', dnn2_param=None):
        """
        combine the Ensemble features with shape 64 and
        HoG features processed by PCA as new input.
        :param paras: [parameter of ResNet, of vgg]
        :param dnn1_param: parameter of DNN of Ensemble
        :param device:
        :param dnn2_param: parameter of DNN of Ensemble2
        """
        super(Ensemble2, self).__init__()
        self.device = device
        dnn1 = Ensemble(paras, device, dnn1_param)
        dnn1.dnn = dnn1.dnn[:-4]
        print(dnn1.dnn)
        self.en = dnn1
        dnn2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )
        if dnn2_param:
            dnn2.load_state_dict(torch.load(dnn2_param, map_location=device))
            for param in dnn2.parameters():
                param.requires_grad = False
        self.dnn = dnn2.to(device)
        self.hog = Extract_features(batch_size=64)
        print(198)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        # res_out = self.en.res(x)  # output.shape(batch_size, 256)
        # vgg_out = self.en.vgg(x)  # output.shape(batch_size, 256)
        # inputs = torch.cat((res_out, vgg_out), 1)  # 512
        en_out = self.en(x)  # output.shape(batch_size, 64)
        hog_features = self.hog.extract_pca(x)  # output.shape(batch_size, batch_size)64
        new_inputs = torch.cat((en_out, torch.from_numpy(hog_features).to(self.device)),
                               1)
        return self.dnn(new_inputs)


class Ensemble256(nn.Module):
    def __init__(self, paras, dnn1_param, device='cpu', dnn2_param=None):
        """
        combine the Ensemble features with shape 256 and
        HoG features processed by PCA as new input.
        :param paras: [parameter of ResNet, of vgg, of HoG]
        :param dnn1_param: parameter of DNN of Ensemble
        :param device:
        :param dnn2_param: parameter of DNN of Ensemble256
        """
        super(Ensemble256, self).__init__()
        self.device = device
        dnn1 = Ensemble(paras[:2], device, dnn1_param)
        dnn1.dnn = dnn1.dnn[:-6]
        # print(dnn1.dnn)
        self.en = dnn1
        self.hog = Hog_dnn512(device, paras[2])
        self.hog.model = self.hog.model[:-2]
        dnn2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )
        if dnn2_param:
            dnn2.load_state_dict(torch.load(dnn2_param, map_location=device))
            for param in dnn2.parameters():
                param.requires_grad = False
        self.dnn = dnn2.to(device)
        print(198)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        en_out = self.en(x)  # output.shape(batch_size, 256)
        hog_out = self.hog(x)
        new_inputs = torch.cat((en_out, hog_out), 1)
        return self.dnn(new_inputs)


class Ensemble512(nn.Module):
    def __init__(self, paras, device='cpu', dnn_param=None, ):
        """
        combine the Ensemble features with shape 512 and
        HoG features processed by PCA as new input.
        :param paras: [parameter of ResNet, of vgg, of HoG]
        :param device:
        :param dnn_param: parameter of DNN of Ensemble512
        """
        super(Ensemble512, self).__init__()
        self.res = Pretrained_res50(device, paras[0])
        self.res.model.fc = self.res.model.fc[:-2]  # output 256
        self.vgg = Pretrained_vgg(device, paras[1])
        self.vgg.model.classifier[6] = self.vgg.model.classifier[6][:-2]  # output 256
        self.hog = Hog_dnn512(device, paras[2])
        self.hog.model = self.hog.model[:-6]
        dnn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )
        if dnn_param:
            # dnn.load_state_dict(torch.load('./dataset/xray_dnn_parameters.pkl'))
            dnn.load_state_dict(torch.load(dnn_param, map_location=device))
            for param in dnn.parameters():
                param.requires_grad = False
        self.dnn = dnn.to(device)
        print(198)

    def forward(self, x):  # [batch_size, channel:1, height, width]
        res_out = self.res(x)
        vgg_out = self.vgg(x)
        inputs_512 = torch.cat((res_out, vgg_out), 1)
        hog_out = self.hog(x)
        inputs = torch.cat((inputs_512, hog_out), 1)
        return self.dnn(inputs)


class Hog_dnn512(nn.Module):
    def __init__(self, device='cpu', param=None,):
        """
        A DNN based on HoG of data/images
        :param device:
        :param param: parameter of DNN
        """
        super(Hog_dnn512, self).__init__()
        dnn = nn.Sequential(
            nn.Linear(5408, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )
        if param:
            dnn.load_state_dict(torch.load(param, map_location=device))
            for param in dnn.parameters():
                param.requires_grad = False
        self.model = dnn.to(device)
        self.hog = Extract_features(batch_size=64)
        self.device = device

    def forward(self, x):
        hog_features = torch.from_numpy(self.hog.extract_hog(x)).to(self.device)
        return self.model(hog_features)

