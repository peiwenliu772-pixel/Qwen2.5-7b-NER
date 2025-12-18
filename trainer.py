# train.py
import os
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import swanlab

from config_loader import load_config
from model import QwenLoRANER
from data_process import load_qwen_ner_data
from utils import set_seed, evaluate

def train_one_epoch(myconfig, model, loader, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss = 0.0
    step = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if myconfig.use_swan and step % myconfig.swan_log_steps == 0:
            log_data = {
                "train/loss": loss.item(),
                "gpu/mem_allocated_GB": torch.cuda.memory_allocated() / 1024**3,
                "gpu/mem_reserved_GB": torch.cuda.memory_reserved() / 1024**3
            }
            swanlab.log(log_data)

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)

def main():

    myconfig = load_config("qwen_ner_config/qlora_config.json")
    print(f"当前数据集: {myconfig.dataset}")
    print(f"加载模型: {myconfig.pretrained_model_name}")
    set_seed(myconfig.seed)

    os.makedirs(myconfig.output_dir, exist_ok=True)
    log_dir = os.path.join(myconfig.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    myconfig.pretrained_model_path,
    trust_remote_code=True,
    padding_side="left" 
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_loader, dev_loader, _ = load_qwen_ner_data(myconfig, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenLoRANER(myconfig).to(device)

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": myconfig.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=myconfig.learning_rate)
    total_steps = len(train_loader) * myconfig.epochs
    warmup_steps = int(total_steps * myconfig.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # swanlab
    if myconfig.use_swan:
        swanlab.init(project=myconfig.swan_project, name=myconfig.swan_run_name)


    best = -1.0
    for ep in range(1, myconfig.epochs + 1):
        train_loss = train_one_epoch(myconfig, model, train_loader, optimizer, scheduler)
        dev_metrics = evaluate(myconfig, model, tokenizer, dev_loader)
        swanlab.log({
            "epoch": ep,
            "train/epoch_loss": train_loss,
            "dev/precision": dev_metrics["precision"],
            "dev/recall": dev_metrics["recall"],
            "dev/f1": dev_metrics["f1"],
        })
        print(f"[Epoch {ep}] train_loss={train_loss:.4f}, dev={dev_metrics}")
        log_path = os.path.join(myconfig.output_dir, "train_log.txt")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {ep} | train_loss={train_loss:.4f} | dev={dev_metrics}\n")

        if dev_metrics["f1"] > best:
            best = dev_metrics["f1"]
            model.model.save_pretrained(myconfig.output_dir)
            tokenizer.save_pretrained(myconfig.output_dir)
            print(f"保存最佳模型到 {myconfig.output_dir} (F1={best:.4f})")


if __name__ == "__main__":
    main()
