# 模块 08：工具库 NumPy / Pandas / Matplotlib / sklearn

> 难度 ★★　覆盖：S2-07–11、NC-01–04、NA-05　建议 60 分钟

d2l 用 torch，但考纲（尤其 NOAI C 区）要 NumPy/Pandas/Matplotlib/sklearn。这一份给最小可用集。

## 1. NumPy

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])      # 创建
a.shape, a.dtype                     # (2, 2), int64
a[0, 1], a[:, 1], a[1, :]            # 索引与切片
a.reshape(4), a.T                    # 变形与转置
a + 10, a * 2                        # 向量化运算
a.sum(axis=0), a.mean(), a.std()     # 统计（axis=0 按列）
```

广播：形状从右往左对齐，某维为 1 可扩展。`np.zeros`、`np.ones`、`np.arange`、`np.linspace`、
`np.random.randn` 是常用创建函数。

## 2. Pandas

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

## 4. scikit-learn

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
