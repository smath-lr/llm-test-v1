import pytest
import yaml
from pathlib import Path


@pytest.mark.real_api
def test_full_end2end_flow(llm_client, request):
    """完整端到端链路测试：配置加载 → prompt预处理 → API调用 → 返回断言"""
    # 1.加载yaml配置
    BASE_DIR = Path(__file__).parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["deepseek"]["api_key"] is not None

    # 2.prompt预处理 + API调用
    prompt = "2+3等于多少，请只用数字回答"
    res = llm_client.chat(prompt)

    # ========== 所有记录信息存入stash，给makereport钩子读取 ==========
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": res,
        "response_len": len(res)
    }

    # 3.结果断言
    assert "5" in res


@pytest.mark.real_api
def test_full_multi_turn_flow(llm_client, request):
    """多轮对话端到端测试：验证模型上下文记忆能力"""
    # 1.加载yaml配置
    BASE_DIR = Path(__file__).parent.parent
    config_path = BASE_DIR / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["deepseek"]["api_key"] is not None

    # 构造多轮对话上下文
    messages = [
        {"role": "user", "content": "我的名字是小明"}
    ]
    # 第一轮提问
    res1 = llm_client.chat(messages)
    messages.append({"role": "assistant", "content": res1})
    # 第二轮提问，依赖上文记忆
    messages.append({"role": "user", "content": "我叫什么名字？"})
    res2 = llm_client.chat(messages)

    # stash存入完整对话信息，失败时全部保存到json
    request.node.stash["payload_data"] = {
        "multi_turn_messages": messages,
        "prompt": [messages[0]["content"], messages[-1]["content"]],
        "prompt_len": [len(messages[0]["content"]), len(messages[-1]["content"])],
        "response": [res1, res2],
        "response_len": [len(res1), len(res2)],
    }

    # 断言：模型记住名字，返回包含小明
    assert "小明" in res2