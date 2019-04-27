from flask import Flask, request, Markup, jsonify
import pickle
import numpy as np
import random

app = Flask(__name__)

hash_size = 100000
num_url = 10000
P = 103801
K = 7


def hash_func(x, a, b, R, prime=P):
    return ((a * x + b) % prime) % R


class BloomFilter:
    def __init__(self, table_size, k=K):
        np.random.seed(0)
        self.R = table_size
        self.bit_array = np.zeros(int(table_size))
        self.K = k
        hash_param = np.random.choice(range(P), 2*k, replace=False)
        self.a = hash_param[:k]
        self.b = hash_param[k:]

    def insert(self, key):
        for i in range(self.K):
            hash_value = hash_func(key, self.a[i], self.b[i], self.R)
            self.bit_array[hash_value] = 1

    def query(self, key):
        for i in range(self.K):
            hash_value = hash_func(key, self.a[i], self.b[i], self.R)
            if self.bit_array[hash_value] != 1:
                return 0
        return 1

    def save(self):
        with open("./bf/bloomfilter_K{}_R{}w.pkl".format(self.K, int(self.R / 10000)), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


random.seed(1)
malicious_ratio = 0.1
malicious_urls = random.sample(range(num_url), int(malicious_ratio * num_url))

bloom_filter = BloomFilter(hash_size)
for url in malicious_urls:
    bloom_filter.insert(url)
# print(np.count_nonzero(bloom_filter.bit_array))
bloom_filter.save()

with open("./bf/bloomfilter_K{}_R{}w.pkl".format(K, int(hash_size / 10000)), 'rb') as input:
    bloom_filter = pickle.load(input)


@app.route("/")
def main_page():
    return Markup('<h1>Malicious URL checker</h1>' +
                  '<div>Bloom Filter: \n</div>' +
                  '<div>Number of malicious URLs = {}\n</div>'.format(num_url * malicious_ratio) +
                  '<div>R = {}\n</div>'.format(bloom_filter.R) +
                  '<div>K = {}\n</div>'.format(bloom_filter.K) +
                  '<div>\na = {}\n</div>'.format(str(bloom_filter.a)) +
                  '<div>b = {}\n</div>'.format(str(bloom_filter.b)))


@app.route("/query", methods=["GET", "POST"])
def hashing():
    print(request.form)
    url = int(request.form["url"])
    result = bloom_filter.query(url)
    return jsonify({"malicious": result})
