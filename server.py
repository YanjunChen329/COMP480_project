from flask import Flask, request, Markup, jsonify
import random

app = Flask(__name__)

random.seed(1)
hash_size = 1000
num_url = 10000
malicious_ratio = 0.001
P = 1579


def hash_func(N):
    a = random.randint(0, P - 1)
    b = random.randint(0, P - 1)
    print("a: {}; b: {}".format(a, b))
    return lambda x: ((a * x + b) % P) % N


class BloomFilter:
    def __init__(self, table_size, num_func=1):
        self.N = table_size
        self.bit_array = [0] * int(table_size)
        self.hashing_factory = [hash_func(table_size) for i in range(num_func)]

    def insert(self, key):
        for func in self.hashing_factory:
            hash_value = func(key)
            self.bit_array[hash_value] = 1

    def query(self, key):
        for func in self.hashing_factory:
            hash_value = func(key)
            if self.bit_array[hash_value] != 1:
                return 0
        return 1

malicious_urls = random.sample(range(num_url), int(malicious_ratio * num_url))
bloom_filter = BloomFilter(hash_size, num_func=1)
for url in malicious_urls:
    bloom_filter.insert(url)


@app.route("/")
def main_page():
    return Markup('<h1>Malicious URL checker</h1>' +
                  '<div>malicious urls in the bloom filter: \n{}</div>'.format(str(malicious_urls)))


@app.route("/query", methods=["GET", "POST"])
def hashing():
    print(request.form)
    url = int(request.form["url"])
    result = bloom_filter.query(url)
    return jsonify({"malicious": result})
