# 模块 09：深度学习补缺

> 难度 ★★★　覆盖：NB-20–25、ND-02–10、S2-02/03/15　建议 60 分钟

d2l 会讲 CNN/Transformer/训练流程，这一份补考纲多出来的：优化器细节、混合精度、初始化、
GAN、CV 任务、语音、LLM 部署。

## 1. 优化器：SGD → Adam/AdamW

- **SGD**：$\theta\leftarrow\theta-\eta\nabla L$
- **动量**：累积历史梯度，抑制震荡
- **Adam**：一阶矩（均值）与二阶矩（平方均值）的指数移动平均，自适应学习率
- **AdamW**：Adam + 解耦的权重衰减（weight decay 不混进梯度）

**Adam 的更新式**（记这一组就够）：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

$m$ 是一阶矩（梯度的均值），$v$ 是二阶矩（梯度平方的均值）；分母 $\sqrt{v}$ 让每个参数
的步长按自己的梯度尺度缩放。$\beta_1=0.9$、$\beta_2=0.999$ 是默认值，$\epsilon$ 防除零。

**AdamW 的改动**：Adam 把 weight decay 加进梯度（$g_t\leftarrow g_t+\lambda\theta$），
AdamW 则把衰减单独作用在参数上：$\theta\leftarrow\theta-\eta\lambda\theta-\eta\hat m/\sqrt{\hat v}$。
这样衰减不受自适应学习率影响，正则效果更可控。

## 2. 权重初始化：Xavier 与 He

- **Xavier**：方差按输入/输出维度折中，适合 tanh/sigmoid
- **He**：方差按输入维度放大 2 倍，适合 ReLU

目的：让各层输出的方差保持稳定，避免信号逐层消失或爆炸。

**方差怎么定**：设输入维度 $n_{in}$、输出维度 $n_{out}$，希望输出方差与输入相同。

- Xavier（Glorot）：$\mathrm{Var}(w)=\dfrac{2}{n_{in}+n_{out}}$
- He（Kaiming）：$\mathrm{Var}(w)=\dfrac{2}{n_{in}}$

He 多一个因子 2，因为 ReLU 把一半激活置零，方差要补偿。

## 3. 批量归一化（BN）

对每个小批量的同一通道做"减均值、除标准差"，再乘可学习的 $\gamma$、加 $\beta$：

$$\hat x=\frac{x-\mu_{\mathcal B}}{\sqrt{\sigma_{\mathcal B}^2+\epsilon}},\qquad
y=\gamma\hat x+\beta$$

$\mu_{\mathcal B}$、$\sigma_{\mathcal B}^2$ 是当前批量的均值与方差（训练时），
推理时改用训练过程中累积的滑动平均。$\gamma$、$\beta$ 是可学习参数，让网络自己决定
要不要"还原"归一化。

作用：稳定训练、允许更大学习率、有轻微正则化效果。注意训练与推理的统计量不同。

## 4. 混合精度训练

用 float16/bfloat16 做前向与反传、float32 保存主权重，配合损失缩放（loss scaling）。
好处：省显存、提速；风险：数值下溢/上溢。

## 5. CNN、RNN、GAN

- **CNN**：卷积层提取局部特征，池化降维，全连接分类
- **RNN**：按时间步处理序列，隐藏状态传递历史；LSTM/GRU 用门控缓解长程依赖
- **GAN**：生成器造假样本、判别器辨真假，二者对抗训练。目标是极小极大：

$$\min_G\max_D\;\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z}[\log(1-D(G(z)))]$$

训练时交替更新 $D$ 与 $G$；理想平衡点是 $G$ 生成的分布等于真实分布、$D$ 输出恒为 0.5。

## 6. 计算机视觉任务与模型

| 任务 | 输出 | 代表模型 |
|---|---|---|
| 图像分类 | 类别 | ResNet |
| 语义分割 | 每个像素的类别 | U-Net |
| 目标检测 | 框 + 类别 | YOLO、SSD、DETR |
| 多模态 | 图文匹配/生成 | CLIP、扩散模型 |

**卷积输出尺寸**：输入 $H\times W$、核 $k\times k$、填充 $p$、步长 $s$，
输出 $\lfloor(H+2p-k)/s\rfloor+1$。例：$28\times28$、$k=5$、$p=2$、$s=1$ → $28\times28$。

**感受野**：一层层堆叠后，一个输出单元"看到"的输入区域。两个 $3\times3$ 卷积堆叠的感受野
等于一个 $5\times5$，但参数更少（$2\times9<25$）。

OpenCV 常用：读图 `cv2.imread`、缩放 `cv2.resize`、颜色空间 `cv2.cvtColor`、边缘 `cv2.Canny`。

## 7. 语音

- 信号处理：分帧、加窗、频谱（梅尔谱）
- 语音识别（ASR）：音频 → 文本
- 语音合成（TTS）：文本 → 音频

## 8. LLM 与部署

- 原理：Transformer 解码器、自回归预测下一个词元、预训练 + 指令微调
- 多模态：文本、图像、音频联合建模
- 本地部署：下载开源权重（如 Qwen）、加载 tokenizer 与模型、生成
- API 调用：用 HTTP 请求把 prompt 发给服务（如 OpenRouter），解析返回
- **采样参数**：温度 $T$ 控制随机性（$T\to0$ 趋近贪心，$T$ 大更随机）；
  top-p（核采样）只在累计概率达到 $p$ 的最小词集合里采样；top-k 只在概率最高的 k 个词里采样

## 9. 自测

1. Adam 与 SGD 的核心差别是什么？AdamW 又改了什么？
2. ReLU 网络该用 Xavier 还是 He 初始化？为什么？
3. BN 在训练与推理时的行为有什么不同？
4. 混合精度训练为什么要 loss scaling？
5. 目标检测与图像分类的输出有什么不同？

> [!note]- 答案
> 1. Adam 用一阶/二阶矩自适应每个参数的学习率；AdamW 把权重衰减与梯度解耦。
> 2. He；ReLU 会把一半激活置零，方差要按输入维度放大补偿。
> 3. 训练用当前批量的均值/方差；推理用训练时累积的滑动统计量。
> 4. float16 表示范围小，梯度容易下溢；乘一个缩放因子再反缩放。
> 5. 分类输出类别；检测输出多个边界框及其类别与置信度。
