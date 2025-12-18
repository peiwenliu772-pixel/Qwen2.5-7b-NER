from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./pre_models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "*.safetensors",             
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ],
    resume_download=True
)

print("模型已下载到：", model_path)
