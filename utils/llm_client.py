import logging
import random
import time
from email.utils import parsedate_to_datetime
from typing import Dict, List, Union

from httpx2 import ConnectError, RemoteProtocolError
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from tenacity import RetryCallState, retry, stop_after_attempt, stop_before_delay


# ========== 顶层常量（收集阶段就会加载） ==========
MAX_PROMPT_LENGTH = 200
# 最多尝试 3 次：首次请求 + 重试 2 次
MAX_RETRIES = 2
MAX_ATTEMPTS = MAX_RETRIES + 1
# 本次调用总预算、预算里预留的单次有效请求时长。单次请求超时见 config.yaml 的 timeout
TOTAL_BUDGET_SECONDS = 25
ATTEMPT_SECONDS = 2
# Full Jitter：cap = min(base * 2^attempt, max_wait)，sleep = max(min_sleep, random(0, cap))
RETRY_SLEEP_BASE = 1.0
RETRY_SLEEP_MAX = 8.0
RETRY_SLEEP_MIN = 0.2
# 连接失败里不可重试的 DNS 原因
_DNS_MARKERS = (
    "nodename nor servname provided",
    "name or service not known",
)
# 500 只认响应体 type / code，不看 message 文案
_TEMP_FAULT_CODES = {
    "server_error",
    "internal_server_error",
    "temporarily_unavailable",
    "service_unavailable",
    "overloaded",
    "服务端临时故障",
    "临时故障",
}


def _status_code(exc) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _header(exc, name: str):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get(name)
    if value:
        return value
    return headers.get(name.lower())


def _retry_after_seconds(exc) -> float | None:
    """有 Retry-After 才返回秒数；没有或解析不出则返回 None。"""
    raw = _header(exc, "Retry-After")
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            moment = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if moment is None:
            return None
        seconds = moment.timestamp() - time.time()
    if seconds < 0:
        return None
    return seconds


def _backoff_jitter(attempt_number: int) -> float:
    """Full Jitter。attempt 从 0 起，第一次重试的 cap 等于 base。"""
    attempt = max(attempt_number - 1, 0)
    try:
        cap = RETRY_SLEEP_BASE * (2 ** attempt)
    except OverflowError:
        cap = RETRY_SLEEP_MAX
    cap = min(cap, RETRY_SLEEP_MAX)
    if cap <= RETRY_SLEEP_MIN:
        return RETRY_SLEEP_MIN
    return max(RETRY_SLEEP_MIN, random.uniform(0, cap))


def _is_temporary_server_fault(exc) -> bool:
    """只看 body 的 type、code 是否为服务端临时故障。"""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    for key in ("type", "code"):
        value = body.get(key)
        if isinstance(value, str) and value.strip().casefold() in _TEMP_FAULT_CODES:
            return True
    return False


def _is_retryable_connection(exc) -> bool:
    if not isinstance(exc, APIConnectionError):
        return False
    cause = exc.__cause__
    if not isinstance(cause, (RemoteProtocolError, ConnectError)):
        return False
    text = f"{exc} {cause}".lower()
    return not any(marker in text for marker in _DNS_MARKERS)


def _retry_interval(exc, attempt_number: int) -> float | None:
    """返回本次重试间隔。None 表示该异常不重试。"""
    status = _status_code(exc)
    retry_after = _retry_after_seconds(exc)
    # 429 必须带 Retry-After 才重试，等待时间用该值
    if isinstance(exc, RateLimitError) or status == 429:
        return retry_after
    # 500 只在解析出服务端临时故障时重试，间隔走默认退避
    if status == 500 and isinstance(exc, InternalServerError):
        if not _is_temporary_server_fault(exc):
            return None
        return _backoff_jitter(attempt_number)
    # 502/504 走默认退避；503 有 Retry-After 时用它，否则走默认退避
    if status in (502, 503, 504):
        if status == 503 and retry_after is not None:
            return retry_after
        return _backoff_jitter(attempt_number)
    if _is_retryable_connection(exc):
        return _backoff_jitter(attempt_number)
    return None


# ============ LLM api客户端对象 ============
class LLMClient:
    def __init__(self, api_key, base_url, model, timeout, temperature, max_tokens):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0   # 关闭SDK内部重试，由tenacity统一控制重试
        )
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
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
            # 多轮对话：逐条校验 user prompt，再挂到实例上
            for message in prompt:
                if message.get("role") == "user":
                    content = message.get("content")
                    self._validate_prompt(content)
                    self.biz_logger.info(f"收到prompt，长度={len(content)}")
            self.messages = prompt
        else:
            # 单轮对话：自动组装消息列表
            self._validate_prompt(prompt)
            self.biz_logger.info(f"收到prompt，长度={len(prompt)}")
            self.messages = [{"role": "user", "content": prompt}]
        return self.messages


    # 第一次请求开始前，把截止时间记在本次重试状态上
    @staticmethod
    def _before_attempt(state: RetryCallState):
        if state.attempt_number != 1:
            return
        state.retry_deadline = time.monotonic() + TOTAL_BUDGET_SECONDS


    # 异常可重试，且剩余预算盖得住本次等待加下一次有效请求
    @staticmethod
    def _can_retry(state: RetryCallState) -> bool:
        if state.outcome is None:
            return False
        exc = state.outcome.exception()
        if exc is None or state.attempt_number >= MAX_ATTEMPTS:
            return False
        interval = _retry_interval(exc, state.attempt_number)
        if interval is None:
            return False
        deadline = getattr(state, "retry_deadline", None)
        if deadline is None:
            return False
        remain = deadline - time.monotonic()
        if not (remain > interval + ATTEMPT_SECONDS):
            return False
        state.planned_retry_wait = interval
        return True


    @staticmethod
    def _retry_wait(state: RetryCallState) -> float:
        return getattr(state, "planned_retry_wait", 0)


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
        stop=stop_after_attempt(MAX_ATTEMPTS) | stop_before_delay(TOTAL_BUDGET_SECONDS),
        wait=_retry_wait,
        retry=_can_retry,
        before=_before_attempt,
        before_sleep=_on_retry_sleep,
        reraise=True,
    )
    def chat(self, prompt, timeout=None, temperature=None, max_tokens=None, model=None):
        """
        【调用 DeepSeek，返回模型回复】统一对话入口，兼容两种入参
            :param prompt:
                字符串 → 单轮对话，自动包装成messages
                list → 多轮对话，直接传入完整messages数组
            :return: 模型返回的文本内容
        """
        self._get_chat_messages(prompt)
        resp = self.client.chat.completions.create(
            model=self.model if model is None else model,
            messages=self.messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            timeout=self.timeout if timeout is None else timeout
        )

        # API返回结果校验
        if not resp.choices:
            self.biz_logger.error("API 返回空choices")
            raise RuntimeError("API 返回空choices")
        
        self.biz_logger.info("API 调用成功")
        return resp.choices[0].message.content
    
