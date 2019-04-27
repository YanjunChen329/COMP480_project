import random
import numpy as np
import pickle
import torch

hash_size = 10000
num_url = 10000
num_malicious = 1000
# P = 103801
P = 13099
K = 5


def hash_func(x, a, b, R, prime=P):
    # return ((a * x + b) % prime) % R
    return (a * x + b) % prime


class BloomFilter:
    def __init__(self, table_size, k=K):
        np.random.seed(0)
        self.R = table_size
        self.bit_array = np.zeros(int(table_size))
        self.k = k
        hash_param = np.random.choice(range(P), 2*k, replace=False)
        self.a = hash_param[:k]
        self.b = hash_param[k:]
        print("a: {}".format(str(self.a)))
        print("b: {}".format(str(self.b)))

    def insert(self, key):
        for i in range(self.k):
            hash_value = hash_func(key, self.a[i], self.b[i], self.R)
            self.bit_array[hash_value] = 1

    def query(self, key):
        for i in range(self.k):
            hash_value = hash_func(key, self.a[i], self.b[i], self.R)
            if self.bit_array[hash_value] != 1:
                return 0
        return 1

    def save(self):
        # with open("./bf/bloomfilter_K{}_R{}w.pkl".format(self.k, int(self.R / 10000)), 'wb') as output:
        #     pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        with open("./bf/bloomfilterP_K{}_R{}.pkl".format(self.k, int(self.R)), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


def generate_data(bf, train_ratio=0.8):
    n = 100000
    key = np.arange(0, n, 1).reshape((-1, 1))
    label = np.zeros((n, 1))

    for i in range(n):
        label[i, :] = bf.query(key[i, :])

    print("num of collisions:", np.count_nonzero(label))
    data = np.concatenate((key, label), axis=1)
    np.random.shuffle(data)
    train_size = int(n * train_ratio)
    training_set = data[:train_size, :]
    testing_set = data[train_size:, :]

    # np.savetxt("training_abP.txt", training_set)
    # np.savetxt("testing_abP.txt", testing_set)
    return training_set, testing_set


def generate_adversarial_traffic(bf):
    # net = torch.load("./checkpoint/BFNet2_51")
    net = torch.load("./checkpoint/FCNsine_35")
    print(net)

    input = np.arange(100000, 200000, 1)
    print(input)

    # randomly generating traffic
    # random_count = 0
    # for i in range(input.shape[0]):
    #     random_count += bf.query(input[i])
    # print("collision rate (random): {}".format(random_count / 100000.))

    # use model to generate traffic
    x = input
    # x = np.divide(input - np.mean(training[:, 0]), np.std(training[:, 0]))
    test_x = torch.from_numpy(x.reshape((-1, 1))).type(torch.DoubleTensor).cuda()
    net.eval()
    out = net(test_x)
    _, test_pred = torch.max(out.data, 1)
    np_pred = test_pred.cpu().numpy()
    print(np.count_nonzero(np_pred))

    chosen = test_x[test_pred == 1].cpu().numpy().ravel()
    net_count = 0.
    for i in range(chosen.shape[0]):
        net_count += bf.query(int(chosen[i]))
    print(net_count)
    print("collision rate (net): {}".format(net_count / chosen.shape[0]))



if __name__ == '__main__':
    random.seed(1)
    # malicious_urls = random.sample(range(num_url), num_malicious)

    # bf = BloomFilter(P)
    # for url in malicious_urls:
    #     bf.insert(url)
    # # print(np.count_nonzero(bloom_filter.bit_array))
    # bf.save()

    with open("./bf/bloomfilterP_K{}_R13099.pkl".format(K), 'rb') as input:
        bf = pickle.load(input)

    # for url in malicious_urls:
    #     if bf.query(url) == 0:
    #         print("??")

    training, _ = generate_data(bf)

    generate_adversarial_traffic(bf)


