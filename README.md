## 参考文献

本项目基于 Li 等人（2025）提出的训练免费多风格素描生成框架 [Text to Sketch Generation with Multi-Styles](https://openreview.net/forum?id=C7Ed8V44JY) 进行扩展，实现了语言驱动的风格选择和多风格加权混合功能。请按照该仓库的配置程序完成安装后，将我们的仓库克隆到 `M3S/` 目录下，即可运行。

- **论文：**  
  Tengjie Li, Shikui Tu, Lei Xu. *Text to Sketch Generation with Multi-Styles*. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)*. [OpenReview 链接](https://openreview.net/forum?id=C7Ed8V44JY)

- **官方代码仓库：**  
  [https://github.com/CMACH508/M3S](https://github.com/CMACH508/M3S)

# 多风格迁移功能

## 功能概述

新增两个功能：

1. **自动风格匹配**：输入文本描述，系统自动从风格库匹配最相关的风格图片
2. **多风格混合**：支持任意数量（k种）风格的加权混合，不再限制为2种

## 安装

```bash
pip install sentence-transformers scikit-learn
```

## 使用方法

详细示例见 `example_multi_style.py`

### 方式1：自动风格匹配

设置 `user_query`（文本描述）和 `style_base_dir`（风格库目录），系统自动匹配前 `top_k_styles` 个风格。

### 方式2：直接指定风格

设置 `style_image_paths`（风格图片路径列表）和 `style_weights`（权重列表，可选，默认等权重）。

## 一些参数

- `user_query`：用户查询文本（用于风格匹配）
- `style_base_dir`：风格库目录，包含 style1, style2, ... 等子文件夹
- `top_k_styles`：匹配前k个风格（默认1）
- `style_image_paths`：风格图片路径列表
- `style_weights`：每个风格的权重（自动归一化）
- `per_style_interpolation`：K/V拼接方式
  - `True`：每个风格单独插值（效果更好，但显存占用大，3个风格约需40G）
  - `False`：先融合再插值（显存友好，超过3个风格建议用此方式）

## 一些代码修改

主要修改文件：

- `config.py`：添加多风格配置参数
- `utils/style_matcher.py`：基于语义相似度的风格匹配
- `utils/adain.py`：扩展 AdaIN 支持多风格加权混合
- `utils/latent_utils.py`：支持加载/反演多个风格图片
- `appearance_transfer_model.py`：多风格 AdaIN 回调和注意力处理器
- `run.py`：集成风格匹配和多风格迁移流程

- 
## 实验程序

**对自动风格匹配的测试**
1. 通过VGG对风格仓库中的所有图和输出图片进行风格相似度打分，查看是否VGGloss最低的图片是否是通过风格匹配出来的图片

**对多风格混合的测试**
1. 通过CLIP对输入prompt和多风格输出图片进行相似度打分
2. 通过VGG来对输入的参考风格图和输出图片进行风格相似度打分
3. 通过一个预训练的768→1 的 MLP进行美学打分


