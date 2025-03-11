- [项目概述](#项目概述)
- [我的自制大模型推理框架课程介绍](#我的自制大模型推理框架课程介绍)
- [一，数学和编程基础专栏](#一数学和编程基础专栏)
- [二，神经网络基础部件](#二神经网络基础部件)
- [三，经典卷积神经网络模型](#三经典卷积神经网络模型)
- [四，深度学习炼丹](#四深度学习炼丹)
- [五，深度学习模型压缩](#五深度学习模型压缩)
- [六，模型推理部署](#六模型推理部署)
- [七，进阶课程](#七进阶课程)
- [参考资料](#参考资料)

## 项目概述

本仓库项目是个人总结的计算机视觉和大语言模型学习笔记，包含深度学习基础知识、神经网络基础部件详解、深度学习炼丹策略、深度学习模型压缩算法、深度学习推理框架代码解析及动手实战。

`LLM` 基础及推理优化的专栏笔记请参考 [llm_note](https://github.com/HarleysZhang/llm_note) 仓库。

## 我的自制大模型推理框架课程介绍

1. **框架亮点**：基于 Triton + PyTorch 开发的轻量级、且简单易用的大模型推理框架，采用类 Pytorch 语法的 Triton 编写算子，绕开 Cuda 复杂语法实现 GPU 加速。
2. **价格：499**。非常实惠和便宜，课程、项目、面经、答疑质量绝对对得起这个价格。
3. **课程优势​**：
   - **手把手教你从 0 到 1 实现大模型推理框架**。
   - 项目导向 + 面试导向 + **分类总结的面试题**。
   - 2025 最新的高性能计算/推理框架岗位的大厂面试题汇总
4. **项目优势​**：
	- 架构清晰，代码简洁且注释详尽，覆盖大模型离线推理全流程。​
    - 运用 OpenAI Triton 编写高性能计算 Kernel，开发矩阵乘法内核，效率堪比 cuBLAS。​
    - 依托 PyTorch 进行高效显存管理。​
    - 课程项目完美支持 FlashAttentionV1、V2、V3 与 `GQA`，以及 `PageAttention` 的具体实现。​
    - 使用 `Triton` 编写融合算子，如 KV 线性层融合等。​
    - 适配最新的 `llama/qwen2.5/llava1.5` 模型，相较 transformers 库，在 llama3 1B 和 3B 模型上，加速比最高可达 `4` 倍。
5. **分类总结部分面试题**：

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center"><img src="./images/read_me/interview1.png" width="100%" alt="llava_output2"></td>
    <td align="center"><img src="./images/read_me/interview2.png" width="100%" alt="llava_output"></td>
  </tr>
</table>

6. **项目运行效果**:

`llama3.2-1.5B-Instruct` 模型流式输出结果测试：

![流式输出](./images/read_me/generate.gif)

`Qwen2.5-3B` 模型（社区版本）流式输出结果测试：

![流式输出](./images/read_me/output.gif)

`Llava1.5-7b-hf` 模型流式输出结果测试:

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center"><img src="./images/read_me/llava_output2.gif" width="90%" alt="llava_output2"></td>
    <td align="center"><img src="./images/read_me/llava_output1.gif" width="100%" alt="llava_output"></td>
  </tr>
</table>

感兴趣的同学可以扫码联系课程购买，这个课程是我和[《自制深度学习推理框架》作者](https://space.bilibili.com/1822828582)一起合力打造的，内容也会持续更新优化。

<div align="center">
<img src="./images/read_me/fu_qcode.jpg" width="40%" alt="transformer_block_mp">
</div>

## 一，数学和编程基础专栏

- [深度学习数学基础-概率与信息论](./1-math_ml_basic/深度学习数学基础-概率与信息论.md)
- [深度学习基础-机器学习基本原理](./1-math_ml_basic/深度学习基础-机器学习基本原理.md)
- [随机梯度下降法的数学基础](./1-math_ml_basic/随机梯度下降法的数学基础.md)
- [Python 编程思维导航](./1-math_ml_basic/python_learn_xmind)

## 二，神经网络基础部件

1，**神经网络基础部件**：

1. [神经网络基础部件-卷积层详解](./2-deep_learning_basic/神经网络基础部件-卷积层详解.md)
2. [神经网络基础部件-BN 层详解](./2-deep_learning_basic/神经网络基础部件-BN层详解.md)
3. [神经网络基础部件-激活函数详解](./2-deep_learning_basic/神经网络基础部件-激活函数详解.md)

2，**深度学习基础**：

- [反向传播与梯度下降详解](2-deep_learning_basic/反向传播与梯度下降详解.md)
- [深度学习基础-参数初始化详解](./2-deep_learning_basic/深度学习基础-参数初始化详解.md)
- [深度学习基础-损失函数详解](./2-deep_learning_basic/深度学习基础-损失函数详解.md)
- [深度学习基础-优化算法详解](./2-deep_learning_basic/深度学习基础-优化算法详解.md)

## 三，经典卷积神经网络模型

**1，卷积神经网络的经典 backbone**：

- [ResNet网络详解](3-classic_backbone/ResNet网络详解.md)
- [DenseNet 网络详解](3-classic_backbone/DenseNet论文解读.md)
- [ResNetv2 网络详解](3-classic_backbone/ResNetv2论文解读.md)
- [经典 backbone 网络总结](3-classic_backbone/经典backbone总结.md)

**2，轻量级网络详解**：

- [MobileNetv1论文详解](3-classic_backbone/efficient_cnn/MobileNetv1论文详解.md)
- [ShuffleNetv2论文详解](3-classic_backbone/efficient_cnn/ShuffleNetv2论文详解.md)
- [RepVGG论文详解](3-classic_backbone/efficient_cnn/RepVGG论文详解.md)
- [CSPNet论文详解](3-classic_backbone/efficient_cnn/CSPNet论文详解.md)
- [VoVNet论文解读](3-classic_backbone/efficient_cnn/VoVNet论文解读.md)
- [轻量级模型设计总结](5-model_compression/模型压缩-轻量化网络总结.md)

## 四，深度学习炼丹

1. [深度学习炼丹-数据标准化](./4-deep_learning_alchemy/深度学习炼丹-数据标准化.md)
2. [深度学习炼丹-数据增强](./4-deep_learning_alchemy/深度学习炼丹-数据增强.md)
3. [深度学习炼丹-不平衡样本的处理](./4-deep_learning_alchemy/深度学习炼丹-不平衡样本的处理.md)
4. [深度学习炼丹-超参数设定](./4-deep_learning_alchemy/深度学习炼丹-超参数调整.md)
5. [深度学习炼丹-正则化策略](./4-deep_learning_alchemy/深度学习炼丹-正则化策略.md)

## 五，深度学习模型压缩

1. [深度学习模型压缩算法综述](./5-model_compression/深度学习模型压缩方法概述.md)
2. [模型压缩-轻量化网络设计与部署总结](./5-model_compression/模型压缩-轻量化网络详解.md)
3. [模型压缩-剪枝算法详解](./5-model_compression/模型压缩-剪枝算法详解.md)
4. [模型压缩-知识蒸馏详解](./5-model_compression/模型压缩-知识蒸馏详解.md)
5. [模型压缩-量化算法详解](./5-model_compression/模型压缩-量化算法概述.md)

## 六，模型推理部署

1，模型推理部署：

- [卷积神经网络复杂度分析](./6-model_deploy/卷积神经网络复杂度分析.md)
- [模型压缩部署概述](./6-model_deploy/模型压缩部署概述.md)
- [矩阵乘法详解](./6-model_deploy/卷积算法优化.md)
- [模型推理加速技巧-融合卷积和BN层](./6-model_deploy/模型推理加速技巧-融合卷积和BN层.md)

2，`ncnn` 框架源码解析：

- [ncnn 源码解析-sample 运行](5-model_deploy/ncnn源码解析-sample运行.md)
- [ncnn 源码解析-Net 类](5-model_deploy/ncnn源码解析-Net类.md)

3，异构计算

1. 移动端异构计算：`neon` 编程
2. GPU 端异构计算：`cuda` 编程，比如 `gemm` 算法解析与优化

## 七，进阶课程

1，推荐几个比较好的深度学习模型压缩与加速的仓库和课程资料：

1. [神经网络基本原理教程](https://github.com/microsoft/ai-edu/blob/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.1-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%89%8D%E5%90%91%E8%AE%A1%E7%AE%97%E5%8E%9F%E7%90%86.md)
2. [AI-System](https://microsoft.github.io/AI-System/): 深度学习系统，主要从底层方向讲解深度学习系统等原理、加速方法、矩阵成乘加计算等。
3. [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)：很好的 pytorch 深度学习教程。

2，一些笔记好的博客链接：

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): 国内比较好的博客大都参考这篇文章。
- [C++ 并发编程（从C++11到C++17）](https://paul.pub/cpp-concurrency/): 不错的 C++ 并发编程教程。
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

3，最后，持续高质量创作不易，有 `5` 秒空闲时间的，**可以扫码关注我的公众号-嵌入式视觉**，记录 CV 算法工程师成长之路，分享技术总结、读书笔记和个人感悟。
> 公众号不会写标题党文章，也不输出给大家带来的焦虑的内容！

![qcode](images/others/qcode.png)

4，Star History Chart：

[![Star History Chart](https://api.star-history.com/svg?repos=HarleysZhang/deep_learning_system&type=Date)](https://star-history.com/#HarleysZhang/deep_learning_system&Date)

## 参考资料

- 《深度学习》
- 《机器学习》
- 《动手学深度学习》
- [《机器学习系统：设计和实现》](https://openmlsys.github.io/index.html)
- [《AI-EDU》](https://ai-edu.openai.wiki/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/index.html)
- [《AI-System》](https://github.com/microsoft/AI-System/tree/main/Textbook)
- [《PyTorch_tutorial_0.0.5_余霆嵩》](https://github.com/TingsongYu/PyTorch_Tutorial)
- [《动手编写深度学习推理框架 Planer》](https://github.com/Image-Py/planer)
- [distill：知识精要和在线可视化](https://distill.pub/)
- [LLVM IR入门指南](https://github.com/Evian-Zhang/llvm-ir-tutorial)
- [nanoPyC](https://github.com/vesuppi/nanoPyC/tree/master)
- [ClassifyTemplate](https://github.com/Yale1417/ClassifyTemplate)
- [pytorch-classification](https://github.com/bearpaw/pytorch-classification)
