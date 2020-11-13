#### Bibtex

编译使用分成以下四步：
- 用Latex编译.tex文件，这时会生成一个.aux文件，它将告诉BibTex要使用哪些引用
- 用BibTex编译（注意这里编译的是aux文件，例如上一步编译了a.tex，那么会生成a.aux，在这一步使用命令`bibtex a`进行编译）
- 再次用Latex编译.tex文件，这个时候在文档中已经包含了参考文献，但此时引用的编号很可能不正确。
- 最后一次用Latex编译.tex文件。如果一切顺利的话，这时所有的东西都正常了。

关于引用：
在tex文档`\end{document}`之前，加入
```
bibliographystyle{abbrv}
\bibliography{kkk}
```
这两行，前者`abbrv`应该是表示一种引用格式，后者表示引用的是`kkk.bib`文件。

题外话：之前为了贪图方便，会用texshop一键生成。但是最好别用了，因为它疯狂占内存。。。

