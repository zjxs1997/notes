# 用youdao词典在线查询，可以通过设置命令行快捷执行

import sys
import re
import requests

base_url = 'http://dict.youdao.com/w/'
pattern = '<div class="trans-container">(.+?)</div>'
tele = re.compile(pattern, re.S)
explain_pattern = '<li>(.+?)</li>'
explain_tele = re.compile(explain_pattern, re.S)

base_eng_url = 'http://dict.youdao.com/w/eng/'
eng_pattern = '<span style="cursor: pointer;">(.+?)</span>'
eng_tele = re.compile(eng_pattern, re.S)

content_pattern = '<span class="contentTitle"><a.+?>(.+?)</a>'
content_tele = re.compile(content_pattern, re.S)

def search_word(word):
    word_url = base_url + word
    response = requests.get(word_url)
    html_text = response.text
    explains = tele.findall(html_text)[0]
    print(word)
    for explain in explain_tele.findall(explains):
        print(explain.strip())
    print()


def search_eng_word(word):
    word_url = base_eng_url + word
    response = requests.get(word_url)
    html_text = response.text

    for explain in content_tele.findall(html_text):
        if explain.isalpha():
            return explain
    return '<unk>'

    # explains = tele.findall(html_text)[0]
    # print(word)
    # for explain in explain_tele.findall(explains):
    #     print(explain.strip())
    # print()




if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        print("please input at least one word!")
    else:
        for word in args[1:]:
            search_word(word)



