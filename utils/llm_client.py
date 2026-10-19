import logging 
from pathlib import Path

import yaml 
from openai import OpenAI

# ============ 路径与常量 ============
BASE_DIR = Path(__file__).parent.parent          # 项目根目录
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_PROMPT_LENGTH = 200


# ============ 业务日志（独立，不依赖测试框架） ============
biz_logger = logging.getLogger("llm_business")
biz_logger.setLevel(logging.INFO)
if not biz_logger.handlers:   # 业务模块只加载一次，防止重复挂 handler
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # 控制台输出handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    biz_logger.addHandler(stream_handler)

    # 文件输出handler
    file_handler = logging.FileHandler(LOG_DIR / "llm_client.log", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    biz_logger.addHandler(file_handler)

    
# ============ 加载配置 ============
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"配置文件不存在！路径：{CONFIG_PATH}")

deepseek_cfg = config["deepseek"]
client = OpenAI(
    api_key=deepseek_cfg["api_key"],
    base_url=deepseek_cfg["base_url"]
)


# ============ 核心调用 ============
def chat(prompt, model=deepseek_cfg["model"], temperature=0.7, max_tokens=500):
    """
    【调用 DeepSeek，返回模型回复】统一对话入口，兼容两种入参
    :param prompt:
        字符串 → 单轮对话，自动包装成messages
        list → 多轮对话，直接传入完整messages数组
    :return: 模型返回的文本内容
    """
    if isinstance(prompt, list):
        # 多轮对话：直接使用传入的messages
        messages = prompt
    else:
        # 单轮对话：自动组装消息列表
        _validate_prompt(prompt)
        biz_logger.info(f"收到prompt，长度={len(prompt)}")
        messages = [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # API返回结果校验
    if not resp.choices:
        biz_logger.error("API 返回空choices")
        raise RuntimeError("API 返回空choices")
    
    biz_logger.info("API 调用成功")
    return resp.choices[0].message.content


def _validate_prompt(prompt: str) -> None:
    """校验 prompt 输入合法性，非法则抛 ValueError。"""
    if prompt is None:
        biz_logger.error("参数校验失败：prompt 为 None")
        raise ValueError("prompt不能为None")   
     
    if len(prompt.strip()) == 0:
        biz_logger.error("参数校验失败：prompt 为 空字符串")
        raise ValueError("prompt不能为空")
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        biz_logger.error(f"参数校验失败：prompt 超长，长度={len(prompt)}")
        raise ValueError(f"prompt长度不能超过{MAX_PROMPT_LENGTH}")