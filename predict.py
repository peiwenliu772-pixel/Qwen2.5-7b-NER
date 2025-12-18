# predict.py
import torch
from utils import evaluate
from config_loader import load_config
from model import QwenLoRANER
from data_process import load_qwen_ner_data
from transformers import AutoTokenizer

def main():
    myconfig = load_config("qwen_ner_config/lora_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
    myconfig.pretrained_model_path,
    trust_remote_code=True,
    padding_side="left" 
    )
    _, _, test_loader = load_qwen_ner_data(myconfig, tokenizer)
    model = QwenLoRANER(myconfig, adapter_path=myconfig.output_dir).to(device)
    test_metrics = evaluate(myconfig, model, tokenizer, test_loader)
    print(f"测试集结果: {test_metrics}")

if __name__ == "__main__":
    main()
