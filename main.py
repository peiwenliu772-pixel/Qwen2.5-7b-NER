# main.py
import os
import json
import random


import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_process import load_qwen_ner_data
from model import QwenLoRANER
from config_loader import load_config
from tqdm import tqdm
import swanlab


# -----------------------------
# 0) 随机数
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 1) 工具：从 JSON 文本解析实体
# -----------------------------
def parse_prediction_to_entities(pred_text: str):
    
    # 期望模型输出：
    #   [{"name": "...", "type": "GENE"}, ...]

    pred_text = pred_text.strip()
    if not pred_text:
        return set()

    # 尝试从像 "```json ... ```" 这种格式中抠出 JSON
    if "```" in pred_text:
        parts = pred_text.split("```")
        # 简单找最长的那段
        candidate = max(parts, key=len)
        pred_text = candidate.strip()

    try:
        data = json.loads(pred_text)
    except Exception:
        # 如果不是标准 JSON，直接放弃这条（记 0 个实体）
        return set()

    if isinstance(data, dict):
        data = [data]

    entities = set()
    if not isinstance(data, list):
        return set()

    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("entity") or item.get("text")
        etype = item.get("type", "GENE")
        if not name:
            continue
        entities.add((name.lower().strip(), etype))
    return entities


# -----------------------------
# 2) 计算 P / R / F1
# -----------------------------
def getPRF(pred_list, true_list):
    TP, pre, true = 0, 0, 0
    for pred_entities, true_entities in zip(pred_list, true_list):
        TP += len(pred_entities & true_entities)
        pre += len(pred_entities)
        true += len(true_entities)
    precision = TP / pre if pre != 0 else 0.0
    recall = TP / true if true != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0
    return precision, recall, f1


# -----------------------------
# 3) 评估
# -----------------------------
@torch.no_grad()
def evaluate(myconfig, model, tokenizer, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_was_training = model.training
    model.eval()

    max_new_tokens = myconfig.max_new_tokens

    pred_list, true_list = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=True):
        sentences = batch["sentences"]
        gold_entities_batch = batch["entities"]  

        for sent, gold_entities in zip(sentences, gold_entities_batch):
            # 构造推理时的指令（只到 user，不带 gold）
            user_prompt = (
                "你是一个生物医学实体识别模型，请从下面的句子中抽取所有基因/蛋白相关实体。\n"
                "请只输出 JSON 数组，每个元素形如：{\"name\": 实体文本, \"type\": \"GENE\"}。\n"
                f"句子：{sent}"
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert biomedical named entity recognition model."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            output_ids = model.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,       
                top_p=None,
                top_k=None,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            # 只取生成部分
            gen_ids = output_ids[0, inputs.size(1):]
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred_entities = parse_prediction_to_entities(pred_text)

            gold_set = {
                (e["name"].lower().strip(), e["type"])
                for e in gold_entities
            }

            pred_list.append(pred_entities)
            true_list.append(gold_set)

    precision, recall, f1 = getPRF(pred_list, true_list)

    if model_was_training:
        model.train()
    return {"precision": precision, "recall": recall, "f1": f1}


# -----------------------------
# 4) 训练一个 epoch
# -----------------------------
def train_one_epoch(myconfig, model, loader, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss = 0.0
    step =0
    for batch in tqdm(loader, desc="Training", leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        step += 1
        if myconfig.use_swan and step % myconfig.swan_log_steps == 0:
            log_data = {"train/loss": loss.item()}
            log_data["gpu/mem_allocated_GB"] = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            log_data["gpu/mem_reserved_GB"] = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            swanlab.log(log_data)

        total_loss += loss.item() * input_ids.size(0)
       
    return total_loss / len(loader.dataset)


# -----------------------------
# 5) 主流程
# -----------------------------
def main():
    #读取配置
    myconfig = load_config("qwen_ner_config/lora_config.json")
    print(f"当前数据集: {myconfig.dataset}")
    print(f"加载模型: {myconfig.pretrained_model_name}")
    set_seed(myconfig.seed)
    #输出目录
    myconfig.output_dir = os.path.join("output", myconfig.dataset, myconfig.pretrained_model_name + myconfig.method)
    os.makedirs(myconfig.output_dir, exist_ok=True)
    log_path = os.path.join(myconfig.output_dir, "train_log.txt")
    
    # 是否使用 QLoRA
    use_qlora = myconfig.use_qlora
    if use_qlora:
        print("使用 QLoRA 微调")
    else:
        print("使用 LoRA 微调")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------  tokenizer  --------
    tokenizer = AutoTokenizer.from_pretrained(
        myconfig.pretrained_model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --------  DataLoader  --------
    train_loader, dev_loader, test_loader = load_qwen_ner_data(myconfig, tokenizer)
    print(f"训练集样本数：{len(train_loader.dataset)}")
    print(f"单 epoch 训练步数：{len(train_loader)}")

    # --------  模型（Qwen + LoRA/QLORA）  --------
    if use_qlora:
        print("正在加载 4-bit QLoRA 模型")
        model = QwenLoRANER(myconfig).to(device)  
    else:
        print("加载标准 LoRA 模型")
        model = QwenLoRANER(myconfig).to(device)

    # --------  只训练 LoRA 参数 分组weight decay  --------
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": myconfig.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=myconfig.learning_rate)

    # --------  scheduler  --------
    total_steps = len(train_loader) * myconfig.epochs
    warmup_steps = int(total_steps * myconfig.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # --------  swanlab 初始化  --------
    if myconfig.use_swan:
        swanlab.init(
            project=myconfig.swan_project,
            name=myconfig.swan_run_name
        )


    # --------  训练 & 验证  --------
    best = -1.0
    for ep in range(1, myconfig.epochs + 1):
        train_loss = train_one_epoch(myconfig, model, train_loader, optimizer, scheduler)
        dev_metrics = evaluate(myconfig, model, tokenizer, dev_loader)
        swanlab.log({
                    "epoch": ep,
                    "dev/precision": dev_metrics["precision"],
                    "dev/recall": dev_metrics["recall"],
                    "dev/f1": dev_metrics["f1"],
                    "train/loss_epoch": train_loss
                })
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} dev={dev_metrics}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {ep} | train_loss={train_loss:.4f} | dev={dev_metrics}\n")
        
        # --------  保存最佳模型  --------
        main_score = dev_metrics.get("f1", 0.0)
        if main_score > best:
            best = main_score
            model.model.save_pretrained(myconfig.output_dir)  
            tokenizer.save_pretrained(myconfig.output_dir)
            print(f"保存最佳adapter到 {myconfig.output_dir} (F1={best:.4f})")

    # --------  测试集评估  --------
    print("加载最佳模型进行测试")
    model = QwenLoRANER(myconfig).to(device)
    test_metrics = evaluate(myconfig, model, tokenizer, test_loader)
    print(f"测试结果: {test_metrics}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"测试结果: {test_metrics}\n")


if __name__ == "__main__":
    main()
