## 关于浏览器的部分

- window是全局域，同时也表示浏览器窗口，可以通过`innerWidth`和`innerHeight`属性活的窗口的内部宽高。当然也可以获取外部。

- navigator对象可以获取浏览器的信息，包括名称、版本、语言等等。

- screen对象可以获取屏幕的信息。

- location对象获取当前页面的url信息。`location.assign`可以加载一个新的页面，而`location.reload`可以刷新当前页面

- document对象表示当前页面，也就是DOM树的根节点。用document查找DOM树的节点。document还有一个cookie属性。

- 操作dom。dom的操作主要是更新、遍历、添加和删除。要操作DOM，首先就得获得相应的。js中可以通过三个get函数获得，分别是id（返回的是唯一的）、CSS（返回的是array）和TagName（也是array）。通过children、firstElementChild和last...可以获得DOM的子节点。parentElement可以获得父节点。

    - 更新DOM，有两种方式。一是直接修改目标DOM的`innerHTML`属性，这种方式可以插入新的html标签。另一种是修改`textContent`属性，通过这个方式修改，如果要强行插入html标签也会被编码掉。此外，还可以通过DOM的`style`属性修改css。
    - 插入DOM。用`appendChild`，可以把一个节点添加到父节点的最后一个子节点之后。这个节点可以是原本在DOM树中的，这种情况会把原始的节点删掉，放到新的地方，也可以是通过`document.createElement(标签名称)`的方式创建的。用`insertBefore(newEle, refEle)`可以把节点添加到参考节点之前。
    - 删除DOM。需要有被删除的节点以及它的父节点，通过父节点的`removeChild`方法删除。不过节点虽然删除了，但是仍在内存中。另外，通过children属性删除多个节点的时候，需要注意，这个属性的值是在动态变化的。
    
- 关于表单，我的理解是用`<form>`标签包起来一堆标签，有些是`<input>`标签，可以指定type，来定义用户输入的不同类型数据；也可以用`<button>`标签，把type设置成submit就可以点击的时候提交表单。还可以给form绑定一个onsubmit函数，用来检查提交的数据是否符合规则。
因为表单里也是DOM树，所以可以通过js轻松获得这些节点，并且通过value或checked得到/设置表单控件的值。
还有个type为file的控件，允许上传文件，但是从这个控件获取的value值是不可靠的。

- AJAX，大概还算比较熟悉吧，就是异步地获取数据，XMLHttpRequest对象。通过onreadystatechange回调函数判断相应结果、以及做出相应的回应。默认情况下，js的ajax请求的url域名必须和当前页面完全一致，包括域名、协议，大部分还要保证端口号也一致。

- promise。创建promise对象的时候，需要传入一个函数，这个函数接受两个参数resolve和reject，分别表示成功和失败的时候的回调函数。这个函数内部的逻辑随便怎么写，只要保证成功和失败的时候调用回调函数的逻辑即可（当然不调用也无所谓）。创建了这个promise对象之后，可以调用它的then和catch方法。这两个方法的参数都是一个函数，这个函数会分别被绑定到resolve或reject上。也可以套娃，一层一层then。还有Promise.all与.race的用法，分别是执行n个promise、让n个promise竞争，这之后通过then执行某个回调函数。






