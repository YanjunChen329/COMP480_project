import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy():
    FCN_train = np.loadtxt("./checkpoint/FCN_train_acc.txt")
    FCN_train = np.mean(FCN_train.reshape((100, -1)), axis=1)
    FCN_test = np.loadtxt("./checkpoint/FCN_test_acc.txt")

    FCNsine_train = np.loadtxt("./checkpoint/FCNsine_train_acc.txt")
    FCNsine_train = np.mean(FCNsine_train.reshape((100, -1)), axis=1)
    FCNsine_test = np.loadtxt("./checkpoint/FCNsine_test_acc.txt")

    BF_train = np.loadtxt("./checkpoint/BFNet2_train_acc.txt")
    BF_train = np.mean(BF_train.reshape((100, -1)), axis=1)
    BF_test = np.loadtxt("./checkpoint/BFNet2_test_acc.txt")

    x = np.arange(0, 100)
    plt.plot(x, FCN_train, color="red", linestyle="-", label="FCN training")
    plt.plot(x, FCN_test, color="red", linestyle="--", label="FCN testing")
    plt.plot(x, FCNsine_train, color="blue", linestyle="-", label="FCN(sine) training")
    plt.plot(x, FCNsine_test, color="blue", linestyle="--", label="FCN(sine) testing")
    plt.plot(x, BF_train, color="green", linestyle="-", label="BF_Net training")
    plt.plot(x, BF_test, color="green", linestyle="--", label="BF_Net testing")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def plot_loss():
    FCN_loss = np.loadtxt("./checkpoint/FCN_train_loss.txt")
    FCN_loss = np.mean(FCN_loss.reshape((100, -1)), axis=1)

    FCNsine_loss = np.loadtxt("./checkpoint/FCNsine_train_loss.txt")
    FCNsine_loss = np.mean(FCNsine_loss.reshape((100, -1)), axis=1)

    BF_loss = np.loadtxt("./checkpoint/BFNet2_train_loss.txt")
    BF_loss = np.mean(BF_loss.reshape((100, -1)), axis=1)

    x = np.arange(0, 100)
    plt.plot(x, FCN_loss, color="red", linestyle="-", label="FCN")
    plt.plot(x, FCNsine_loss, color="blue", linestyle="-", label="FCN(sine)")
    plt.plot(x, BF_loss, color="green", linestyle="-", label="BF_Net")
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # plot_accuracy()
    plot_loss()
