import os

import pytest

from utils.config_loader import MASKED_API_KEY, load_llm_config


@pytest.mark.real_api
def test_tc088_masked_api_key_passthrough(llm_client, llm_config):
    """TC-088：配置中的 api_key 是打码值，请求头携带环境变量中的原文。"""
    header = llm_client.client.auth_headers.get("Authorization")
    env_api_key = os.getenv("DEEPSEEK_API_KEY")
    header_ok = (
        bool(env_api_key)
        and env_api_key != MASKED_API_KEY
        and header == f"Bearer {env_api_key}"
    )
    assert llm_config["api_key"] == MASKED_API_KEY
    assert header_ok, "请求头应携带环境变量中的密钥"


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