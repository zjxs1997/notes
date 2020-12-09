## JavaScript xjb记

- js的字符串里有个反引号，\`，这个可以用来写python中三个引号的换行字符串，也可以写类似于f-string的东西。

```javascript
var s = `dsfsd
sdfsd
sdf`;
console.log(s);

var name = 'tom';
var hello =`hello ${name}`;
console.log(hello);
```

- 常用的数据结构，有Array、Map和Set，作用相当于python中的list、dict和set，只不过有些使用方式和函数名称不一样，需要习惯一下。

- 遍历元素有for-in、for-of和forEach三种方式。其中
```javascript
var a = ['a', 'b', 'c'];
for (var i in a) console.log(i);
// in 打印的是0，1，2；如果a是set或dict，则什么都不会打印
for (var i of a) console.log(i);
// of 打印的是a，b，c；如果a是set也会打印abc

var b = new Map([[1, 'a'], [2, 'b'], [3, 'c']])
for (var i of a) console.log(i[0], i[1]);
// 依次打印1a 2b 3c
```
而forEach是通过array之类的调用的，传入一个函数，其参数为element指向当前元素的值，index指向当前索引，array指向调用的array本身。



（中间省略一大堆。。。）


- Array的sort方法会直接修改原始的array。另外用它排序数字的时候，它会默认用元素转换成的string比较。可以传入一个函数，参数是比较的对象x y，如果`x<y`则返回-1，大于返回1，相等返回0，升序排序。

- 原型链 proto，blabla

