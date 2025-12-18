# 📖 基于 Qwen2.5-7B 的指令微调实体识别（NER）
## 🚀 一、项目简介
本项目演示了如何使用LoRA/QLoRA 技术对Qwen2.5-7B-Instruct模型进行指令微调（Instruction Tuning），以完成生物医学命名实体识别（NER）任务。项目实现了从数据预处理到模型评估的完整流程，并集成 **SwanLab** 进行训练过程可视化。

## 📊 二、数据集来源：
bc2gm命名实体识别数据集。

下载地址：https://github.com/spyysalo/bc2gm-corpus?utm_source=chatgpt.com
## 🚀 三、环境与文件结构
### 项目结构
```
qwen_ner
├── README.md
├── config_loader.py
├── data
│   └── bc2gm1
├── data_process.py
├── download_model.py
├── model.py
├── output
│   └── bc2gm1
│       ├── qwen_ner_LoRA
│       └── qwen_ner_QLoRA
├── pre_models
│   └── Qwen2.5-7B-Instruct
├── predict.py
├── qwen_ner_config
│   ├── lora_config.json
│   └── qlora_config.json
├── swanlog
├── trainer.py
└── utils.py
```
###  依赖安装

```
pip install transformers peft  bitsandbytes torch 
pip install swanlab tqdm numpy
```
## 🚀四、快速开始
### 训练练/评估
直接运行trainer.py即可。
```
python main.py
```
### 测试

直接运行predict.py即可。
```
python predict.py
```
## 五、实验结果
### 📊 微调方法性能对比

| 微调方法 | Batch Size | 显存占用 (近似) | F1 Score (参考) |
| :---: | :---: | :---: | :---: |
| **LoRA (BF16)** | 4 | ~25GB | 82.7% |
| **QLoRA (4-bit)** | 4 | ~13 GB | 82.1% |