import requests
import json

server = "http://127.0.0.1:5000"


def query_url(url):
    r = requests.post("http://127.0.0.1:5000/query", data={"url": url})
    content = json.loads(r.text)
    print(url, content["malicious"])
    return content["malicious"]

if __name__ == '__main__':
    for u in range(1000):
        query_url(u)
