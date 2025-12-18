import json

class Arguments:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_config(path):
    """读取配置文件并封装为 Arguments 对象"""
    with open(path, "r", encoding="utf-8") as f:
        args = json.load(f)
    return Arguments(**args)
