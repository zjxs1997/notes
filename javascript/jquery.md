## jQuery

要在浏览器里使用jQuery的代码，只需在head的script中引入即可。

### 选择器。
jquery最入门的使用。
- 基础 可以按id查找`a = $('#id')`，按tag查找`$('p')`，按class查找`$('highlight')`。

- 按属性查找`$('[name=email]')`。如果属性的值有特殊字符，需要用双引号引起来。

- tag与class、属性这些都可以组合起来查找，表示一个and的关系。或者可以用逗号隔开，查找or条件。

- 几个条件之间用空格隔开，就是层级选择器。比如`$('ul p')`就表示在ul的子节点中查找p标签。如果不是用空格而是用>隔开，则是子查找器，后一个必须是前一个的直属子节点。

- 过滤器。都是冒号后面跟个单词，可以限制更多的挑选条件，比如（未）被选中的checkbox、可以输入的标签等。主要还是要熟练。

- 用jquery查找到的jQuery对象还可以调用find方法进一步查找，也可以用filter方法过滤。

- 节点也可以调用parent、next和prev方法得到父节点和sibling节点。

### 修改DOM
选中了元素之后，下一步肯定是要做操作。可以通过text和html方法获取文本或html代码，如果调用的时候传入了一个参数，那就是修改的意思。
要修改css属性则可以调用css方法，这个方法返回的还是jQuery对象，有点像c++里的cin。

用hide与show函数设置是否隐藏。

用prop方法获取/修改元素的属性值（比如CheckBox的是否选中等），而对于表单元素，则统一可以使用val方法获取/修改对应的value属性。

添加元素的方法是append，和原始的那种比较接近。

### 事件
这个用来控制各种事件的处理方式，比如点击某个按钮的时候如何处理等。

基本的方式就是jQuery对象.事件名(处理事件的函数)。

不过这些绑定处理函数的代码不应直接放在head里。因为在执行这些js代码的时候，网页可能还没有完全加载完成，所以可能无法绑定上。正确的做法是，把这些代码放在`$(document).ready(function() {});`中。这也是个事件，意味着网页加载完成，再调用那个函数，在那个函数中绑定各种处理函数。

### AJAX
jQuery中提供了比较丰富的ajax调用。可以直接通过$.ajax进行，也可以直接$.get或$.post，甚至还有$.getJSON。
不详细展开了，，

### 自定义jQuery扩展
有的时候要多次对某些jQuery对象执行相同的代码（比如要设置同一套css）。这种时候就可以通过$.fn自定义扩展。比如
```javascript
$.fn.highlight1 = function () {
    // this已绑定为当前jQuery对象:
    this.css('backgroundColor', '#fffceb').css('color', '#d85030');
    return this;
}
```
这样jQuery对象就可以调用highlight1方法了。
最后的这个return this的写法，也是为了可以保持原始jQuery的风格。

如果需要保存扩展函数的默认参数值，可以保存在`$.fn.<pluginName>.defaults`中，用户可以传入一个object以覆盖一些默认值。



