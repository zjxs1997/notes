import hashlib
import json
import time

import requests


def translate(q):
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

    app_id = '20200717000520803'
    private_key = 'XlvWra57hTKq3Ncc59ee'

    from_lang = 'en'
    to_lang = 'zh'
    salt = 'kkk'

    sign = hashlib.md5((app_id+q+salt+private_key).encode("utf-8")).hexdigest()

    d = {
        "q": q,
        "from": from_lang,
        "to": to_lang,
        "appid": app_id,
        "salt": salt,
        "sign": sign
    }

    r = requests.get(url, d)
    text = r.text

    j = json.loads(text)

    return j


queries = [
    "hello world!\nFuck you!",
    "This is a pencil.",
    "That is an apple.",
]


for q in queries:
    j = translate(q)
    print(j)
    # {'from': 'en',
    #  'to': 'zh',
    #  'trans_result': [{'src': 'Hello world!', 'dst': '你好，世界！'}]}
    print(j['trans_result'][0]['dst'])
    time.sleep(1)


