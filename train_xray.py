from build_xray_dataset import Xray_dataset
from build_xray_models import Resnet, Pretrained_res50, Pretrained_vgg, \
    Ensemble, Ensemble2, Extract_features, Hog_dnn512, Ensemble512, Ensemble256
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_xray(model, name, trainset, testset, epochs=50, batch_size=32, lr=0.001):
    train_num, test_num = len(trainset[1]), len(testset[1])
    xray_trainset = Xray_dataset(trainset, is_train=True)
    xray_testset = Xray_dataset(testset, is_train=False)

    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    PREVIOUS = 0
    ITERATIONS = 0
    loss_dict = {}
    epoch_acc = []
    for epoch in range(1, epochs+1):
        train_right = 0
        test_right = 0
        xray_trainloader = DataLoader(xray_trainset, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
        xray_testloader = DataLoader(xray_testset, batch_size=batch_size, shuffle=True,
                                     pin_memory=True, drop_last=True)
        for i, (train_data, train_label) in enumerate(xray_trainloader):
            train_data = train_data.to(device)
            train_label = train_label.type(torch.LongTensor).to(device)
            ITERATIONS += 1

            pred = model(train_data)
            train_loss = CEloss(pred, train_label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_right += torch.sum(torch.argmax(pred, dim=1) == train_label)
            if ITERATIONS == 1 or ITERATIONS % 10 == 0:
                print("iters: {:d}, train loss = {:.3f}".format(ITERATIONS, train_loss))
                loss_dict[ITERATIONS] = train_loss.item()

        for j, (test_data, test_label) in enumerate(xray_testloader):
            test_data = test_data.to(device)
            test_label = test_label.type(torch.LongTensor).to(device)

            with torch.no_grad():
                pred = model(test_data)
                test_right += torch.sum(torch.argmax(pred, dim=1) == test_label)

        print("epoch: {:d}, train accuracy = {:.3f}%, test accuracy = {:.3f}%".format(epoch,
                                                                                      100 * train_right.item() / train_num,
                                                                                      100 * test_right.item() / test_num))
        epoch_acc.append((100 * train_right.item() / train_num, 100 * test_right.item() / test_num))
        if 100 * test_right.item() / test_num > PREVIOUS:
            PREVIOUS = 100 * test_right.item() / test_num
            torch.save(model.model.state_dict(), './dataset/xray_' + name +
                       '_e' + str(epochs) + '_b' + str(batch_size) +
                       '_lr' + str(lr)[2:] + "_para.pt",
                       _use_new_zipfile_serialization=False)
            with open('./dataset/xray_' + name +
                      '_e' + str(epochs) + '_b' + str(batch_size) +
                      '_lr' + str(lr)[2:] + '_loss_dict.pkl', 'wb') as f:
                pickle.dump(loss_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            if test_right.item() / test_num >= 0.95:
                break
    np.save('./dataset/xray_' + name +
            '_e' + str(epochs) + '_b' + str(batch_size) +
            '_lr' + str(lr)[2:] + '_acc.npy', np.array(epoch_acc))


def train_xray_ensemble(ensemble, trainset, testset, epochs=50, batch_size=64, lr=0.001):
    train_num, test_num = len(trainset[1]), len(testset[1])
    xray_trainset = Xray_dataset(trainset, is_train=True)
    xray_testset = Xray_dataset(testset, is_train=False)

    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble.dnn.parameters(), lr=lr, weight_decay=0.001)

    PREVIOUS = 0
    ITERATIONS = 0
    epoch_acc = []
    for epoch in range(1, epochs+1):
        train_right = 0
        test_right = 0
        xray_trainloader = DataLoader(xray_trainset, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
        xray_testloader = DataLoader(xray_testset, batch_size=batch_size, shuffle=True,
                                     pin_memory=True, drop_last=True)
        for i, (train_data, train_label) in enumerate(xray_trainloader):
            train_data = train_data.to(device)
            train_label = train_label.type(torch.LongTensor).to(device)
            ITERATIONS += 1

            pred = ensemble(train_data)
            train_loss = CEloss(pred, train_label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_right += torch.sum(torch.argmax(pred, dim=1) == train_label)
            if ITERATIONS == 1 or ITERATIONS % 10 == 0:
                print("iters: {:d}, train loss = {:.3f}".format(ITERATIONS, train_loss))

        for j, (test_data, test_label) in enumerate(xray_testloader):
            test_data = test_data.to(device)
            test_label = test_label.type(torch.LongTensor).to(device)

            with torch.no_grad():
                pred = ensemble(test_data)
                test_right += torch.sum(torch.argmax(pred, dim=1) == test_label)

        print("epoch: {:d}, train accuracy = {:.3f}%, test accuracy = {:.3f}%".format(epoch,
                                                                                      100 * train_right.item() / train_num,
                                                                                      100 * test_right.item() / test_num))
        epoch_acc.append((100 * train_right.item() / train_num, 100 * test_right.item() / test_num))
        if 100 * test_right.item() / test_num >= PREVIOUS:
            PREVIOUS = 100 * test_right.item() / test_num
            torch.save(ensemble.dnn.state_dict(), './dataset/xray_dnn' +
                       '_e' + str(epochs) + '_b' + str(batch_size) +
                       '_lr' + str(lr)[2:] + "_para.pkl",
                       _use_new_zipfile_serialization=False)
            if test_right.item() / test_num >= 0.95:
                break
    np.save('./dataset/xray_dnn' +
            '_e' + str(epochs) + '_b' + str(batch_size) +
            '_lr' + str(lr)[2:] + '_acc.npy', np.array(epoch_acc))


if __name__ == '__main__':
    print(77)
    # a = np.array([[2, 3], [4, 5]])
    # b = np.array([2, 2], )

    trainset, testset = [None, None], [None, None]
    trainset[0] = np.load('./dataset/xray/x_train.npy', )[:200]
    trainset[1] = np.load('./dataset/xray/y_train.npy', )[:200]
    testset[0] = np.load('./dataset/xray/x_test.npy', )[:100]
    testset[1] = np.load('./dataset/xray/y_test.npy', )[:100]

    en256 = Ensemble256(['./dataset/xray_res_e50_b64_lr0001_para.pt',  # model.model.state_dict()
                         './dataset/xray_vgg_e50_b64_lr0001_para.pt',  # model.model.state_dict()
                         './dataset/xray_hog_e30_b64_lr0001_para.pt'],
                        './dataset/xray_dnn_e50_b64_batchn_l2_para.pt',
                        device, )
    train_xray_ensemble(en256, trainset, testset, epochs=50, batch_size=64, lr=0.0001)
    '''
    en512 = Ensemble512(['./dataset/xray_res_e50_b64_lr0001_para.pt',  # model.model.state_dict()
                        './dataset/xray_vgg_e50_b64_lr0001_para.pt',  # model.model.state_dict()
                         './dataset/xray_hog_e30_b64_lr0001_para.pt'],
                        device, )
    train_xray_ensemble(en512, trainset, testset, epochs=50, batch_size=64, lr=0.0001)
    
    hog_dnn = Hog_dnn512(device=device, )
    train_xray(hog_dnn, 'hog', trainset, testset, epochs=30, batch_size=64, lr=0.0001)
    
    en2 = Ensemble2(['./dataset/xray_res_e50_b64_lr0001_para.pt',
                    './dataset/xray_vgg_e50_b64_lr0001_para.pt'],
                    './dataset/xray_dnn_e50_b64_batchn_l2_para.pt',
                    device)
    # batch_size must be 64
    train_xray_ensemble(en2, trainset, testset, epochs=50, batch_size=64, lr=0.0001)
    
    resnet50 = Pretrained_res50(device, )
    # vgg = Pretrained_vgg(device, )
    train_xray(resnet50, 'res', trainset, testset)
    # train_xray(vgg, 'vgg', trainset, testset)
    
    en = Ensemble(['./dataset/xray_res_e50_b64_lr0001_para.pt',  # model.model.state_dict()
                   './dataset/xray_vgg_e50_b64_lr0001_para.pt'],  # model.model.state_dict()
                  device, )
    train_xray_ensemble(en, trainset, testset, epochs=50, batch_size=64, lr=0.0001)
    '''

    print(156)
    """
    train_num, test_num = len(trainset[1]), len(testset[1])
    # trainset[0] = np.expand_dims(trainset[0], 1)
    xray_trainset = Xray_dataset(trainset, is_train=True)
    xray_trainloader = DataLoader(xray_trainset, batch_size=32, shuffle=True,
                                  pin_memory=True, drop_last=True)

    xray_testset = Xray_dataset(testset, is_train=False)
    xray_testloader = DataLoader(xray_testset, batch_size=32, shuffle=True,
                                 pin_memory=True, drop_last=True)

    # model = None
    # 2D image
    model = Resnet(in_channel=1, num_labels=3).to(device)
    loss_dict = {}
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    CEloss = nn.CrossEntropyLoss()

    PREVIOUS = 0
    iterations = 0
    epoch_acc = []
    for epoch in range(1, 10 + 1):
        train_right = 0
        test_right = 0
        count = 0
        for train_data, train_label in xray_trainloader:
            train_data = train_data.to(device)
            train_label = train_label.type(torch.LongTensor).to(device)
            iterations += 1
            count += 1

            pred = model(train_data)
            # print(trn_label)  0~9torch.argmax(pred, dim=1)
            train_loss = CEloss(pred, train_label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_right += torch.sum(torch.argmax(pred, dim=1) == train_label)
            if iterations == 1 or iterations % 10 == 0:
                print("iters: {:d}, train loss = {:.3f}".format(iterations, train_loss))
                loss_dict[iterations] = train_loss.item()

        count = 0
        for test_data, test_label in xray_testloader:
            test_data = test_data.to(device)
            test_label = test_label.type(torch.LongTensor).to(device)
            count += 1

            with torch.no_grad():
                pred = model(test_data)
                test_right += torch.sum(torch.argmax(pred, dim=1) == test_label)

        print("epoch: {:d}, train accuracy = {:.3f}%, test accuracy = {:.3f}%".format(epoch,
                                                                                      100 * train_right.item() / train_num,
                                                                                      100 * test_right.item() / test_num
                                                                                      ))
        epoch_acc.append((100 * train_right.item() / train_num, 100 * test_right.item() / test_num))
        if 100 * test_right.item() / test_num > PREVIOUS:

            torch.save(model.state_dict(), "parameters.pkl")
            with open('loss_dict.pkl', 'wb') as f:
                pickle.dump(loss_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            if test_right.item() / test_num >= 0.9:
                break
    np.save('./dataset/xray_acc.npy', np.array(epoch_acc))
    """