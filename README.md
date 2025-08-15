# SGPR - 语义引导的原型修正（苹果叶片病害识别）

## 项目结构

```
SGPR-MAIN/
├── dataset/                    # 病害图像数据集
│   ├── Alternaria leaf spot/    # 交链孢叶斑病图像
│   ├── Blossom blight leaves/   # 花腐病叶片图像
│   ├── Brown spot/              # 褐斑病图像
│   ├── Grey spot/               # 灰斑病图像
│   ├── Health/                  # 健康叶片图像
│   ├── Mosaic/                  # 花叶病图像
│   ├── Powdery mildew/          # 白粉病图像
│   ├── Rust/                    # 锈病图像
│   ├── Scab/                    # 疮痂病图像
│   └── dataset.txt              # 数据集元信息文件
├── results/                     # 输出结果目录
├── data_processing.py           # 数据预处理脚本
├── main.py                      # 主程序入口
├── models.py                    # 模型定义文件
├── test.py                      # 测试评估脚本
└── train.py                     # 模型训练脚本
```

## 快速开始

### 运行项目
```bash
python main.py
```
## 文件说明

### `data_processing.py`
- 图像预处理和增强
- 数据集划分（训练集/验证集/测试集）
- 数据标准化处理

### `models.py`
- 病害识别模型架构
- 特征提取网络

### `train.py`
- 模型训练流程
- 损失函数和优化器配置

### `test.py`
- 模型性能评估
- 计算准确率/召回率等指标

### `main.py`
- 程序主入口
- 整合训练和测试流程
- 命令行参数解析

## 数据集说明

数据集包含9类苹果叶片图像：
1. 交链孢叶斑病 (Alternaria leaf spot)
2. 花腐病 (Blossom blight)
3. 褐斑病 (Brown spot)
4. 灰斑病 (Grey spot) 
5. 健康叶片 (Health)
6. 花叶病 (Mosaic)
7. 白粉病 (Powdery mildew)
8. 锈病 (Rust)
9. 疮痂病 (Scab)


