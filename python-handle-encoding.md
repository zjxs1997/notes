#### python处理编码问题

之前时不时地会遇到一些神秘编码的文本，记录一下处理方法

```python
# chardet 自动识别编码
import chardet
fin = open(“xxx.txt", 'rb')
bytes_text = fin.read()
print(chardet.detect(bytes_text))
```

如果chardet也识别不出来编码，那多半是文档前后编码不一致的问题？
可以用`open(‘xxx.txt’, errors=‘ignore’)`这种方式来处理。不过毕竟是下下策。
