import logging
from typing import Dict, List, Union
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed,  retry_if_exception_type
from openai import APIConnectionError, APITimeoutError, OpenAI


# ========== 顶层常量（收集阶段就会加载） ==========
MAX_PROMPT_LENGTH = 200
# 可重试异常集合：超时、连接失败
RETRY_EXC = (APITimeoutError, APIConnectionError)


# ============ LLM api客户端对象 ============
class LLMClient:
    def __init__(self, api_key, base_url, model, timeout):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0   # 关闭SDK内部重试，由tenacity统一控制重试
        )
        self.model = model
        self.timeout = timeout
        self.max_prompt_length = MAX_PROMPT_LENGTH
        # 拿【业务logger llm_business】
        self.biz_logger = logging.getLogger("llm_business")


    # ============ 校验 prompt 输入合法性 ============
    def _validate_prompt(self, prompt: str) -> None:
        """校验 prompt 输入合法性，非法则抛 ValueError。"""
        if prompt is None:
            self.biz_logger.error("参数校验失败：prompt 为 None")
            raise ValueError("prompt不能为None")   
        
        if len(prompt.strip()) == 0:
            self.biz_logger.error("参数校验失败：prompt 为 空字符串")
            raise ValueError("prompt不能为空")
        
        if len(prompt) > self.max_prompt_length:
            self.biz_logger.error(f"参数校验失败：prompt 超长，长度={len(prompt)}")
            raise ValueError(f"prompt长度不能超过{self.max_prompt_length}")


    # ============ 获取chat(LLM调用)的messages ============
    def _get_chat_messages(self, prompt: Union[str, List[Dict]]) -> list:
        if isinstance(prompt, list):
            # 多轮对话：直接使用传入的messages
            messages = prompt
        else:
            # 单轮对话：自动组装消息列表
            self._validate_prompt(prompt)
            self.biz_logger.info(f"收到prompt，长度={len(prompt)}")
            messages = [{"role": "user", "content": prompt}]
        return messages


    # 单独写一个回调静态函数
    @staticmethod
    def _on_retry_sleep(state: RetryCallState):
        if not state.args:
            return
        instance = state.args[0]  # args[0]拿到self实例
        exc = state.outcome.exception()
        exc_msg = getattr(exc, "custom_msg", str(exc)) if exc else ""
        instance.biz_logger.warning(
            f"接口即将重试，第 {state.attempt_number} 次尝试，上一次异常={exc_msg}"
        )
    

    # ============ 调用 DeepSeek，返回模型回复 ============
    @retry(
        stop=stop_after_attempt(3),        # 最多重试2次（原始1次 + 重试2次，合计尝试3次）
        wait=wait_fixed(1),                # 每次重试间隔 1s
        retry=retry_if_exception_type(RETRY_EXC),  # 只在这两类异常才重试a
        before_sleep=_on_retry_sleep,  # 准备重试休眠前打印
        reraise=True,                      # 重试耗尽之后，继续向上抛出异常
    )
    def chat(self, prompt, timeout=None, temperature=0.7, max_tokens=500):
        """
        【调用 DeepSeek，返回模型回复】统一对话入口，兼容两种入参
            :param prompt:
                字符串 → 单轮对话，自动包装成messages
                list → 多轮对话，直接传入完整messages数组
            :return: 模型返回的文本内容
        """
        messages = self._get_chat_messages(prompt)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout if timeout is None else timeout
        )

        # API返回结果校验
        if not resp.choices:
            self.biz_logger.error("API 返回空choices")
            raise RuntimeError("API 返回空choices")
        
        self.biz_logger.info("API 调用成功")
        return resp.choices[0].message.content
    
