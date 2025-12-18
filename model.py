# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,  prepare_model_for_kbit_training, PeftModel



class QwenLoRANER(nn.Module):
    def __init__(self, config, adapter_path=None):
        super().__init__()

        # ===== 1. 加载 base model(LoRA / QLoRA )=====
        if getattr(config, "use_qlora", False):
            print("使用 QLoRA(4-bit)加载 base model")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=True
            )
            base_model.enable_input_require_grads()
        else:
            print("普通 LoRA(BF16)加载 base model")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto",
                trust_remote_code=True
            )
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()

        # ===== 2. 是否加载已有 LoRA adapter =====
        if adapter_path is not None:
            print(f"推理模式：加载 adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False, 
            )
        else:
            print("训练模式")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(base_model, lora_config)

        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }
