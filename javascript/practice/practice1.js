// https://github.com/iszu/web-learning/blob/master/JavaScript%E7%BB%83%E4%B9%A0%E9%A2%98.md


// 3
function countRepeat (arr) {
    var result = {};
    for (var i of arr) {
        if (i in result) {
            result[i] += 1;
        } else {
            result[i] = 1;
        }
    }
    return result;
}

// 4
function camelCase (str) {
    var re = /[a-zA-Z]+/g;
    var result = '';
    while (true) {
        var word = re.exec(str);
        if (word){
            word = word[0];
            if (result.length == 0) {
                result = word.toLowerCase();
            } else {
                result = result.concat(word[0].toUpperCase());
                result = result.concat(word.slice(1).toLowerCase());
            }
        } else break;
    }
    return result;
}

// 5
function firstNonRepeat (str) {
    // 这样效率不高吧，果然还是先count一遍出现次数，再遍历比较好。
    for (var i = 0; i < str.length; ++i) {
        if ((str.lastIndexOf(str[i]) == i) && (str.indexOf(str[i]) == i))
            return str[i];
    }
    return '';
}

// 6
function flatten (arr) {
    var result = new Array();
    for (var ele of arr) {
        if (Array.isArray(ele)) {
            result = result.concat(flatten(ele));
        } else {
            result.push(ele);
        }
    }
    return result;
}

// 7
function isEmptyObject(obj) {
    if (obj) {
        if (obj.constructor === Object && Object.keys(obj).length === 0)
            return true;
    }
    return false;
}

// 8
function unique (arr) {
    // 这样效率也低。
    var result = [];
    for (var i of arr) {
        if (result.indexOf(i) == -1) {
            result.push(i);
        }
    }
    return result;
}

// 9
function reduce (arr, func, initialValue) {
    var result = initialValue;
    for (var ele of arr) {
        result = func(result, ele);
    }
    return result;
}

