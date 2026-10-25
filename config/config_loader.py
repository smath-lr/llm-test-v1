from pathlib import Path
import yaml


# ============ 获取大模型配置信息 ============
def load_llm_config():
    BASE_DIR = Path(__file__).parent.parent          # 项目根目录
    CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在！路径：{CONFIG_PATH}")

    cfg = config["deepseek"]
    return cfg