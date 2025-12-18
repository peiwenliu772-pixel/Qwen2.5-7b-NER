# data_process.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def left_pad_sequence(sequences, padding_value):
    """
    左填充（left padding）
    sequences: List[Tensor(L_i)]
    return: Tensor(B, L_max)
    """
    max_len = max(seq.size(0) for seq in sequences)
    padded = []

    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), padding_value, dtype=seq.dtype)
            seq = torch.cat([pad, seq], dim=0)
        padded.append(seq)

    return torch.stack(padded, dim=0)


class InstructionNERDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_seq_len):
        super().__init__()
        self.data = self.read_json(json_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Qwen 没有 pad_token，用 eos 代替
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def read_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "sentence": item["sentence"],
            "entities": item.get("entities", [])
        }

    def build_messages(self, sentence, entities):
        gold_entities = [
            {"name": e["name"], "type": e["type"]}
            for e in entities
        ]

        user_prompt = (
            "你是一个生物医学实体识别模型，请从下面的句子中抽取所有基因/蛋白相关实体。\n"
            "请只输出 JSON 数组，每个元素形如："
            "{\"name\": 实体文本, \"type\": \"GENE\"}。\n"
            f"句子：{sentence}"
        )

        return [
            {"role": "system", "content": "You are an expert biomedical named entity recognition model."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(gold_entities, ensure_ascii=False)}
        ]

    def collate_fn(self, batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_sentences = []
        all_entities = []

        for item in batch:
            sentence = item["sentence"]
            entities = item["entities"]
            messages = self.build_messages(sentence, entities)

            # system + user + assistant（完整）
            full_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )[0]

            # system + user（用于算 user 长度）
            user_ids = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )[0]
            user_len = user_ids.size(0)

            # 右截断（不动前面的 user）
            if full_ids.size(0) > self.max_seq_len:
                full_ids = full_ids[:self.max_seq_len]

            attention_mask = torch.ones_like(full_ids, dtype=torch.long)

            labels = full_ids.clone()
            labels[:min(user_len, labels.size(0))] = -100

            all_input_ids.append(full_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_sentences.append(sentence)
            all_entities.append(entities)

        # ===== 左填充（关键修改点）=====
        input_ids = left_pad_sequence(
            all_input_ids,
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = left_pad_sequence(
            all_attention_mask,
            padding_value=0
        )
        labels = left_pad_sequence(
            all_labels,
            padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sentences": all_sentences,
            "entities": all_entities
        }


def load_qwen_ner_data(config, tokenizer):
    train_dataset = InstructionNERDataset(
        config.train_path, tokenizer, config.max_seq_len
    )
    dev_dataset = InstructionNERDataset(
        config.dev_path, tokenizer, config.max_seq_len
    )
    test_dataset = InstructionNERDataset(
        config.test_path, tokenizer, config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=train_dataset.collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=dev_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=test_dataset.collate_fn
    )

    return train_loader, dev_loader, test_loader
