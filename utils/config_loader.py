import os
from pathlib import Path
import yaml
from dotenv import load_dotenv


# ============ 获取大模型配置信息 ============
def load_llm_config():
    BASE_DIR = Path(__file__).parent.parent          # 项目根目录
    CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # api_key 只从根目录 .env 读取，不写回 yaml
    load_dotenv(BASE_DIR / ".env")

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在！路径：{CONFIG_PATH}")

    cfg = config["deepseek"]
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("环境变量 DEEPSEEK_API_KEY 未配置")
    cfg["api_key"] = api_key
    return cfg