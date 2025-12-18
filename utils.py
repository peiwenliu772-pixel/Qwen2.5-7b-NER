# utils.py
import json
import torch
import random
import numpy as np
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_prediction_to_entities(pred_text):
    pred_text = pred_text.strip()
    if not pred_text:
        return set()

    if "```" in pred_text:
        parts = pred_text.split("```")
        pred_text = max(parts, key=len).strip()

    if not pred_text.startswith("[") and not pred_text.startswith("{"):
        return set()

    try:
        data = json.loads(pred_text)
    except Exception:
        return set()

    if isinstance(data, dict):
        data = [data]

    entities = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("entity") or item.get("text")
        etype = item.get("type", "GENE")
        if name:
            entities.add((name.lower().strip(), etype))
    return entities

def getPRF(pred_list, true_list):
    TP, pre, true = 0, 0, 0
    for pred_entities, true_entities in zip(pred_list, true_list):
        TP += len(pred_entities & true_entities)
        pre += len(pred_entities)
        true += len(true_entities)
    precision = TP / pre if pre else 0.0
    recall = TP / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

@torch.no_grad()
def evaluate(myconfig, model, tokenizer, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_was_training = model.training
    model.eval()
    torch.cuda.empty_cache()

   
    pred_list, true_list = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        sentences = batch["sentences"]
        gold_entities_batch = batch["entities"]

        # 1. 构造与训练时一致的 Prompt 格式 (System + User)
        prompt_messages_list = []
        for sent in sentences:
        
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
            prompt_messages_list.append(messages)

        # 2. 批量编码并填充
        input_ids = []
        for messages in prompt_messages_list:
            # 编码 system + user 部分，并自动添加 assistant 提示符
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True, # 必须为 True，以便模型知道从哪里开始生成 Assistant 回复
                return_tensors="pt"
            )[0] # (L,)
            input_ids.append(ids)

        # 批量填充（会遵守 tokenizer.padding_side="left" 的设置）
        inputs = tokenizer.pad(
            {'input_ids': input_ids}, 
            padding='longest', # 填充到 batch 内最长长度
            return_tensors="pt"
        ).to(device)

        # 3. 模型生成
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=myconfig.max_new_tokens,
            do_sample=False,       
            use_cache=True,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,       
            top_p=None,
            top_k=None,
        )
        DEBUG_N = 4
        cnt = 0
        # 4. 解码和评估
        for i, sent in enumerate(sentences):
            # 提取生成的 tokens: outputs[i, inputs['input_ids'].size(1):]
            # inputs['input_ids'].size(1) 是 prompt 的长度 (可能包含 padding)
            prompt_len = inputs['input_ids'].size(1)
            gen_ids = outputs[i, prompt_len:]
            
            # 解码预测文本
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
           
            if cnt < DEBUG_N:
                print("=" * 80)
                print(f"[Sentence {cnt}]")
                print("原句:")
                print(sentences[i])
                print("\n模型输出 pred_text:")
                print(repr(pred_text))
                print("\n真实实体 gold_entities:")
                for e in gold_entities_batch[i]:
                    print(f"  - {e['name']} ({e['type']})")
                cnt += 1
            # 解析预测的实体
            pred_entities = parse_prediction_to_entities(pred_text)
            
            # 准备真实的实体 (转小写和类型)
            gold_set = {(e["name"].lower().strip(), e["type"]) for e in gold_entities_batch[i]}
            
            pred_list.append(pred_entities)
            true_list.append(gold_set)

        del outputs
        torch.cuda.empty_cache()

    precision, recall, f1 = getPRF(pred_list, true_list)
    if model_was_training:
        model.train()
    return {"precision": precision, "recall": recall, "f1": f1}