import os

import pytest

from tests.conftest import MOCK_API_KEY
from utils.config_loader import load_llm_config


@pytest.mark.real_api
def test_tc088_real_api_key_after_chat(llm_client, llm_config, record_payload):
    """TC-088：先完成一次真实调用，再核对此次所用 api_key。"""
    prompt = "只回复ok"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    assert isinstance(result, str) and len(result.strip()) > 0
    env_api_key = os.getenv("DEEPSEEK_API_KEY")
    client_api_key = str(llm_client.client.api_key)
    api_key_ok = (
        bool(env_api_key)
        and client_api_key == env_api_key
        and client_api_key == llm_config["api_key"]
        and client_api_key != MOCK_API_KEY
    )
    assert api_key_ok, "真实 api_key 应透传到客户端"


@pytest.mark.real_api
def test_tc089_single_turn_calculation(llm_client, record_payload):
    """TC-089：单轮计算，回答包含 1035。"""
    prompt = "23乘45等于多少"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    assert "1035" in result


@pytest.mark.real_api
def test_tc090_multi_turn_translation(llm_client, record_payload):
    """TC-090：多轮翻译，回答包含早上好。"""
    messages = [
        {"role": "user", "content": "请把英文翻译成中文"},
        {"role": "assistant", "content": "好的，请给出英文"},
        {"role": "user", "content": "把 good morning 翻译成中文"},
    ]
    record_payload(messages=messages)
    result = llm_client.chat(messages=messages)
    record_payload(messages=messages, response=result)
    assert "早上好" in result


@pytest.mark.parametrize("api_key", [None, ""], ids=["缺失", "空字符串"])
def test_tc091_api_key_missing_or_empty(monkeypatch, api_key):
    """TC-091：加载配置时密钥缺失或为空，不创建客户端，不发请求。"""
    real_getenv = os.getenv

    def getenv(name, default=None):
        if name == "DEEPSEEK_API_KEY":
            return api_key
        return real_getenv(name, default)

    monkeypatch.setattr("utils.config_loader.os.getenv", getenv)
    calls = {"openai": 0, "request": 0}

    def track_openai(*args, **kwargs):
        calls["openai"] += 1
        raise AssertionError("不应创建客户端")

    def track_request(*args, **kwargs):
        calls["request"] += 1
        raise AssertionError("不应发起请求")

    monkeypatch.setattr("src.llm_client.OpenAI", track_openai)
    monkeypatch.setattr(
        "openai.resources.chat.completions.Completions.create",
        track_request,
    )
    with pytest.raises(ValueError, match="环境变量 DEEPSEEK_API_KEY 未配置"):
        load_llm_config()
    assert calls == {"openai": 0, "request": 0}