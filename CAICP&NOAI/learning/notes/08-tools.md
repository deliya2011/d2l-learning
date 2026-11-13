# 模块 08：工具库 NumPy / Pandas / Matplotlib / sklearn

> 难度 ★★　覆盖：S2-07–11、NC-01–04、NA-05　建议 60 分钟

d2l 用 torch，但考纲（尤其 NOAI C 区）要 NumPy/Pandas/Matplotlib/sklearn。这一份给最小可用集。

## 1. NumPy

> 前置：d2l **B02 数据操作**（张量形状、索引、广播；NumPy 与它一一对应）。

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])      # 创建
a.shape, a.dtype                     # (2, 2), int64
a[0, 1], a[:, 1], a[1, :]            # 索引与切片
a.reshape(4), a.T                    # 变形与转置
a + 10, a * 2                        # 向量化运算
a.sum(axis=0), a.mean(), a.std()     # 统计（axis=0 按列）
```

**广播规则**（与 torch 相同）：形状从右往左逐维比较，相等或其中一个为 1 才能运算；
为 1 的那维会被"复制"扩展。例：`(3,1)` 与 `(1,4)` → `(3,4)`。

**axis 的理解**：`axis=0` 是"沿着行方向压下去"，结果少一维、长度等于列数；
`axis=1` 沿列方向压，结果长度等于行数。记不住就写 `a.sum(axis=0).shape` 试一次。

## 2. Pandas

> 前置：d2l **B02 数据预处理**（`read_csv`、`fillna`、`get_dummies`）。

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head(), df.shape, df.columns, df.dtypes
df["col"], df[["a", "b"]], df[df["a"] > 0]     # 选列、筛选
df.loc[0, "a"], df.iloc[0, 1]                   # 按标签 / 按位置
df["a"].fillna(df["a"].mean())                  # 缺失值填补
df.dropna(), df.drop_duplicates()               # 删缺失、去重
df.groupby("class")["score"].mean()             # 分组统计
pd.concat([df1, df2])                           # 追加
df1.merge(df2, on="id", how="left")             # 连接
```

Series 是一列，DataFrame 是二维表。

**merge 的四种连接**：

| how | 保留什么 |
|---|---|
| `inner` | 两表键的交集 |
| `left` | 左表全部，右表匹配不上填 NaN |
| `right` | 右表全部 |
| `outer` | 两表键的并集 |

**groupby 的本质**是"拆分—应用—合并"：按某列分组，对每组算统计量，再拼回表。

## 3. Matplotlib

```python
import matplotlib.pyplot as plt

plt.plot(x, y, label="train")        # 折线
plt.scatter(x, y, s=10)              # 散点
plt.bar(names, values)               # 柱状
plt.pie(values, labels=names)        # 饼图
plt.xlabel("epoch"); plt.ylabel("loss")
plt.legend(); plt.title("curve")
plt.show()
```

要点：多序列要 `label` + `legend`；坐标轴范围 `plt.xlim/ylim`；刻度 `plt.xticks`。

**两种接口**：`plt.*` 是"当前图"的快捷方式，适合快速画；`fig, ax = plt.subplots()`
是面向对象接口，`ax.plot`、`ax.set_xlabel`，多子图时更清晰。

## 4. scikit-learn

> 前置：d2l **B06 线性回归** 与 **B07 softmax 回归**（`fit`/`predict` 的语义）。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)          # 训练
pred = model.predict(X_test)         # 预测
model.score(X_test, y_test)          # 评分（回归 R²，分类准确率）
```

常见模型：`LogisticRegression`、`KNeighborsClassifier`、`SVC`、`DecisionTreeClassifier`、
`RandomForestClassifier`、`KMeans`、`PCA`。统一接口：`fit` / `predict` / `score`。

**Pipeline** 把预处理与模型串起来，避免"训练集标准化用训练集统计量、测试集又用测试集统计量"
这种泄漏：

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)      # 标准化只 fit 训练集，测试集用同一组参数
```

## 5. jieba

```python
import jieba
list(jieba.cut("河道水位预警"))      # ['河道', '水位', '预警']
jieba.lcut(text)                     # 直接返回列表
```

## 6. 自测

1. `a = np.array([[1,2],[3,4]])`，`a.sum(axis=0)` 是什么？
2. Pandas 里 `df.loc` 与 `df.iloc` 的区别？
3. 画两条曲线并加图例，需要哪两个调用？
4. sklearn 模型的三个统一方法是什么？
5. `jieba.lcut` 返回什么类型？

> [!note]- 答案
> 1. `[4, 6]`（按列求和）。
> 2. `loc` 按标签，`iloc` 按位置。
> 3. 每条线 `label=`，最后 `plt.legend()`。
> 4. `fit`、`predict`、`score`。
> 5. 列表（词元组成的 list）。
