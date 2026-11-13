# 模块 03：Python 基础补缺

> 难度 ★★　覆盖：J1-10/11/12、J2-07/08/09、S1-09/10、NA-03/04/05/06　建议 60 分钟

d2l 假设你会这些；考纲会直接考。每节都配最小例子。

## 1. 数制转换

```python
bin(10)        # '0b1010'
oct(10)        # '0o12'
hex(10)        # '0xa'
int("1010", 2) # 10，第二个参数是进制
int("FF", 16)  # 255
```

二进制转十进制：按位权相加，$1010_2=8+2=10$。十进制转十六进制：除 16 取余、倒序。

## 2. 递归

```python
def fact(n):
    if n <= 1:        # 终止条件
        return 1
    return n * fact(n - 1)
```

要点：**终止条件 + 规模缩小**。每次调用压一层栈帧（保存参数与返回地址），太深会
`RecursionError`（默认上限约 1000 层）。Python **没有尾递归优化**，尾递归也会压栈，
所以深递归要改写成显式栈或循环。

斐波那契的朴素递归会重复计算，复杂度指数级；加缓存（`functools.lru_cache`）或改递推。

## 3. 模块与包

```python
import math                 # 整个模块
from random import randint  # 只拿一个函数
import os
```

常用函数：

| 模块 | 函数 | 作用 |
|---|---|---|
| `math` | `sqrt`、`pi`、`gcd` | 数学运算 |
| `random` | `random()`、`randint(a,b)`、`choice(seq)`、`shuffle(lst)` | 随机数与打乱 |
| `os` | `listdir`、`path.exists`、`getcwd`、`makedirs` | 文件系统 |
| `sys` | `argv`、`exit` | 命令行参数与退出 |

`if __name__ == "__main__":` 让"被导入时不执行测试代码"。

**模块搜索路径**：`sys.path` 决定 `import` 去哪里找。当前目录、标准库、第三方包（`site-packages`）。
**包**是带 `__init__.py` 的目录，可以用 `包.模块` 的形式导入。

## 4. 元组、字典、集合

```python
t = (1, 2, 3)         # 元组：不可变，可作字典键
d = {"a": 1}          # 字典：键值对，按键查
s = {1, 2, 3}         # 集合：不重复，支持交并差
```

| 结构 | 有序 | 可变 | 适用 |
|---|---|---|---|
| 列表 | 是 | 是 | 顺序数据、可重复 |
| 元组 | 是 | 否 | 固定记录、字典键 |
| 字典 | 是（3.7+ 保插入序） | 是 | 按键查找、计数 |
| 集合 | 否 | 是 | 去重、交并差 |

字典访问不存在的键会抛 `KeyError`；用 `d.get(k, 默认值)` 更安全。

**字典为什么快**：底层是哈希表，按键的哈希值定位，平均 $O(1)$。所以键必须**可哈希**
（不可变的：字符串、数字、元组；列表不能当键）。

## 5. lambda 与 map / filter / reduce

```python
list(map(lambda x: x * 2, [1, 2, 3]))          # [2, 4, 6]
list(filter(lambda x: x > 1, [1, 2, 3]))       # [2, 3]
from functools import reduce
reduce(lambda a, b: a + b, [1, 2, 3])          # 6
```

`map`、`filter` 返回的是**迭代器**，要 `list()` 才看得到列表；`reduce` 需要导入。

**闭包的坑**：lambda 捕获的是**变量本身**，不是当时的值。

```python
fs = [lambda x: x + i for i in range(3)]
print([f(0) for f in fs])   # [2, 2, 2]，不是 [0, 1, 2]
```

因为三个 lambda 共享同一个 `i`，循环结束时 `i=2`。要固定值就写 `lambda x, i=i: x + i`。

## 6. 列表推导式

```python
[x * x for x in range(4)]              # [0, 1, 4, 9]
[x for x in range(6) if x % 2 == 0]    # [0, 2, 4]
{x: x + 1 for x in range(3)}           # {0: 1, 1: 2, 2: 3}
```

## 7. 自测

1. `int("FF", 16)` 是多少？`hex(255)` 是什么？
2. 写一个递归求 `1+2+...+n` 的函数，终止条件是什么？
3. `random.seed(3)` 的作用是什么？
4. `d = {"a": 1}`，`d["b"]` 与 `d.get("b", 0)` 各会怎样？
5. `list(filter(lambda x: x % 2 == 0, range(5)))` 的结果？

> [!note]- 答案
> 1. 255；`'0xff'`。
> 2. `if n == 0: return 0`，其余 `n + f(n-1)`。
> 3. 固定随机数序列，使结果可复现。
> 4. `d["b"]` 抛 `KeyError`；`d.get("b", 0)` 返回 0。
> 5. `[0, 2, 4]`。
