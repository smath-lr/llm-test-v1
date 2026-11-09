import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# 读到真实密钥后立刻换成这个打码值，下游只拿得到它
MASKED_API_KEY = "***"


def _read_masked_api_key():
    """读取环境变量中的密钥。缺失或空字符串抛错，有效时只返回打码值。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or api_key == "":
        raise ValueError("环境变量 DEEPSEEK_API_KEY 未配置")
    return MASKED_API_KEY


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
    # 对外只放入打码值，原始密钥不离开 _read_masked_api_key
    cfg["api_key"] = _read_masked_api_key()
    return cfg
