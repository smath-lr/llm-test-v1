import pytest
from openai import APIError
from utils.llm_client import chat


# 用例1：真实请求deepseek，使用conftest里的fixture
def test_api_normal_response(valid_prompt, request):
    """使用fixture传入预设prompt，校验API返回数据类型是字符串"""
    res = chat(valid_prompt)
    
    # ========== 所有记录信息存入stash，给makereport钩子读取 ==========
    request.node.stash["payload_data"] = {
        "prompt": valid_prompt,
        "prompt_len": len(valid_prompt),
        "response": res,
        "response_len": len(res)
    }
    assert isinstance(res, str)


# 用例2：mock模拟接口，**不联网，不消耗token**
def test_api_empty_choices(mocker, valid_prompt, request):
    """mock：API返回空choices，预期抛出RuntimeError，不消耗token"""
    # mock底层openai的create方法，模拟返回空choices
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_create.return_value.choices = []

    # ========== 所有记录信息存入stash，给makereport钩子读取 ==========
    request.node.stash["payload_data"] = {
        "prompt": valid_prompt,
        "prompt_len": len(valid_prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "API 返回空choices"
    }
    # 预期调用chat时，抛出RuntimeError
    with pytest.raises(RuntimeError, match="API 返回空choices")as exc_info:
        chat(valid_prompt)


#用例3：mock模拟服务500报错
def test_api_500_error(mocker, valid_prompt, request):
    """mock：API服务500内部异常，预期抛出APIError，不联网"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # side_effect：mock被调用时，抛出APIError，模拟后端500故障
    mock_create.side_effect = APIError(message="服务器内部错误", request=None, body={"code": 500})

    # ========== 所有记录信息存入stash，给makereport钩子读取 ==========
    request.node.stash["payload_data"] = {
        "prompt": valid_prompt,
        "prompt_len": len(valid_prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "服务器内部错误"
    }

    with pytest.raises(APIError, match="服务器内部错误")as exc_info:
        chat(valid_prompt)