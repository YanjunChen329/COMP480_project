import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import network


############# Hyper Parameters #############
LR = 0.001
EPOCH = 300
batchsize = 100


############# Generate data points #############
a = 80
b = 48
K = 107
hash_func = lambda x: (a * x + b) % K

train_x = np.arange(0, 1000)
train_y = np.array(list(map(hash_func, train_x)))
# print(train_x, train_y)

training_loader = []
for i in range(int((train_x.shape[0] + batchsize - 1) / batchsize)):
    if (i+1) * batchsize > train_x.shape[0]:
        x_batch = train_x[i * batchsize:]
        y_batch = train_y[i * batchsize:]
    else:
        x_batch = train_x[i * batchsize:(i+1) * batchsize]
        y_batch = train_y[i * batchsize:(i+1) * batchsize]
    training_loader.append((torch.from_numpy(x_batch.reshape(-1, 1)).type(torch.FloatTensor),
                            torch.from_numpy(y_batch).type(torch.FloatTensor)))

# for i in training_loader:
#     print(i[0].shape, i[1].shape)


############## Loss and Optimizer ################
guess_K = float(np.max(train_y) + 1)
guess_b = float(train_y[0])
guess_a = float(guess_b)
net = network.HashNet(guess_a=guess_a, guess_b=guess_b, guess_K=guess_K)
print(net)
print()
# print(net.parameters())

optimizer = optim.SGD(net.parameters(), lr=LR)
criterion = nn.MSELoss()


################### Training ####################
loss_arr = []
acc_arr = []
test_acc_arr = []

for epoch in range(EPOCH):
    net.train()
    for iteration, (x, y) in enumerate(training_loader):
        output = net(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        print("a={}, b={}, K={}".format(net.a, net.b, net.Modlayer.K.item()))

        y_pred = np.array([round(x.item()) for x in output])
        train_accuracy = (y_pred == y.data).sum().item() / y.data.shape[0]
        loss_arr.append(loss.data)
        acc_arr.append(train_accuracy)
        print('Epoch: ', epoch, '| Iteration: %d' % iteration, '| train loss: %.4f' % loss.data,
              '| train accuracy: %.3f' % train_accuracy)
    # exit()

    # Testing
    # net.eval()
    # out = net(testing_x)
    # _, test_pred = torch.max(out.data, 1)
    # test_accuracy = (test_pred == testing_y.data).sum().item() / testing_y.shape[0]
    # test_acc_arr.append(test_accuracy)
    # print('Epoch: ', epoch, '| test accuracy: %.3f' % test_accuracy)


def plot_loss_accuracy():
    ite = range(len(loss_arr))
    plt.plot(ite, loss_arr)
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss over iterations")
    plt.show()

    plt.plot(ite, acc_arr, label="training")
    plt.plot(ite, test_acc_arr, label="testing")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Loss over iterations")
    plt.show()

# plot_loss_accuracy()
