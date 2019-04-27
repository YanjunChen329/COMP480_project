import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import network
import sys


############# Hyper Parameters #############

LR = 0.01
EPOCH = 100
batchsize = 128


############# Generate random points #############
def normalize(x):
    return np.divide(x - np.mean(x), np.std(x))

# random_url = np.random.randint(0, 1000, (1000, 1)).astype(float) / 1000.
# random_res = np.random.randint(0, 2, 1000)
# random_url = np.arange(0, 1000, 1, dtype=float).reshape((-1, 1)) / 1000.
# random_res = np.zeros(1000, dtype="int64")
# random_res[500:] = 1

# print(random_url, random_res)
# training_loader = [(torch.from_numpy(random_url), torch.from_numpy(random_res))]


################ Actual Dataset ################
train_file = "training_abP.txt"
test_file = "testing_abP.txt"
training_set = np.loadtxt(train_file)
testing_set = np.loadtxt(test_file)
# print(training_set.shape)
# print(np.count_nonzero(training_set, axis=0))
# print(testing_set.shape)
# print(np.count_nonzero(testing_set, axis=0))

def generate_dataloader(data_set, normal=True):
    if normal:
        data_set[:, 0] = normalize(data_set[:, 0])

    loader = []
    for i in range(int((data_set.shape[0] + batchsize - 1) / batchsize)):
        if (i+1) * batchsize > data_set.shape[0]:
            train_batch = data_set[i * batchsize:, :]
        else:
            train_batch = data_set[i * batchsize:(i+1) * batchsize, :]
        loader.append((torch.from_numpy(train_batch[:, 0].reshape(-1, 1)), torch.from_numpy(train_batch[:, 1])))
    return loader


def oversampling_dataloader(data_set, NS_ratio=0.5, normal=True):
    if normal:
        data_set[:, 0] = normalize(data_set[:, 0])

    loader = []
    data_1 = data_set[data_set[:, 1] == 1]
    data_0 = data_set[data_set[:, 1] == 0]
    # print(data_1.shape, data_0.shape)
    batchsize_1 = int(NS_ratio * batchsize)
    batchsize_0 = batchsize - batchsize_1
    for i in range(int((data_0.shape[0] + batchsize_0 - 1) / batchsize_0)):
        if (i+1) * batchsize_0 > data_0.shape[0]:
            train_batch_0 = data_0[i * batchsize_0:, :]
        else:
            train_batch_0 = data_0[i * batchsize_0:(i+1) * batchsize_0, :]
        rand_index = np.random.choice(data_1.shape[0], batchsize_1, replace=False)
        train_batch_1 = data_1[rand_index]
        train_batch = np.concatenate((train_batch_1, train_batch_0), axis=0)
        np.random.shuffle(train_batch)
        # print(train_batch_0, train_batch_1)
        # print(train_batch)
        loader.append((torch.from_numpy(train_batch[:, 0].reshape(-1, 1)), torch.from_numpy(train_batch[:, 1])))
    return loader

# training_loader = generate_dataloader(training_set)
training_loader = oversampling_dataloader(training_set, normal=True)

testing_x = torch.from_numpy(testing_set[:, 0].reshape(-1, 1)).cuda()
testing_y = torch.from_numpy(testing_set[:, 1]).type(torch.LongTensor).cuda()


############### Loss and Optimizer ################
net = network.FullyConnected().cuda()
# net = network.BF_Net().cuda()

optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss().cuda()


################### Training ####################
loss_arr = []
acc_arr = []
test_acc_arr = []
best_acc = 0

print("Start Training")
for epoch in range(EPOCH):
    net.train()
    correct = 0
    total = 0
    for iteration, (x, y) in enumerate(training_loader):
        x = x.cuda()
        y = y.type(torch.LongTensor).cuda()

        output = net(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # grad3 = net.fc3.weight.grad
        # print(np.linalg.norm(grad3.numpy()))
        # if epoch == 100:
        #     print(output.data)

        _, y_pred = torch.max(output.data, 1)
        # print(np.count_nonzero(y_pred))
        # print(y_pred)
        correct += (y_pred == y.data).sum().item()
        total += y.data.shape[0]
        train_accuracy = correct / total
        loss_arr.append(loss.data)
        acc_arr.append(train_accuracy)
        sys.stdout.write("\rEpoch: {} | Iteration: {} | train loss: {:.4f} | train accuracy: {:.4f} ({}/{})"
                         .format(epoch, iteration, loss.data, train_accuracy, correct, total))
        sys.stdout.flush()

    # Testing
    net.eval()
    out = net(testing_x)
    _, test_pred = torch.max(out.data, 1)
    test_accuracy = (test_pred == testing_y.data).sum().item() / testing_y.shape[0]
    test_acc_arr.append(test_accuracy)
    print('\nEpoch: ', epoch, '| test accuracy: %.3f' % test_accuracy)

    # Save Checkpoint
    if test_accuracy > best_acc or epoch == EPOCH - 1:
        torch.save(net, "./checkpoint/FCN_{}".format(epoch))
        best_acc = test_accuracy
        print("Model Saved")

    np.savetxt("./checkpoint/FCN_train_acc.txt", acc_arr)
    np.savetxt("./checkpoint/FCN_train_loss.txt", loss_arr)
    np.savetxt("./checkpoint/FCN_test_acc.txt", test_acc_arr)

