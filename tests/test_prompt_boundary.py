import pytest
from utils.llm_client import MAX_PROMPT_LENGTH


@pytest.mark.real_api
@pytest.mark.parametrize("prompt, should_raise", [
    # ("",  True),                                # 边界：长度0
    ("a",  False),                               # 边界：长度1
    ("a"*2,  False),                             # 边界：长度2
    ("a"*(MAX_PROMPT_LENGTH-1), False),              # 边界：长度199
    ("a"*(MAX_PROMPT_LENGTH), False),                # 边界：长度200
    ("a"*(MAX_PROMPT_LENGTH+1), True)               # 边界：长度201
])
def test_prompt_boundary(llm_client, prompt, should_raise, request):
    # ========== 所有记录信息存入stash，给makereport钩子读取 ==========
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": None,
        "expected_exception": None
    }
    if should_raise:
        with pytest.raises(ValueError, match=r"^prompt长度不能超过") as exc_info:
            llm_client.chat(prompt)
        # 成功捕获预期异常，更新异常信息
        request.node.stash["payload_data"]["expected_exception"] = str(exc_info.value)
    else:
        res = llm_client.chat(prompt)
        request.node.stash["payload_data"].update({
            "response": res,
            "response_len": len(res)
        })
        assert res is not None
        assert len(res) > 0
