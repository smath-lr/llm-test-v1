import logging
import random
import time
from email.utils import parsedate_to_datetime

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
    """只认 APIConnectionError 本身。子类 APITimeoutError 不重试。"""
    if type(exc) is not APIConnectionError:
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


    # ============ 校验字符串 prompt 的内容 ============
    def _validate_prompt(self, prompt: str) -> None:
        """校验字符串 prompt。空内容和超长抛 ValueError。"""
        if len(prompt.strip()) == 0:
            self.biz_logger.error("参数校验失败：prompt 为 空字符串")
            raise ValueError("prompt不能为空")

        if len(prompt) > self.max_prompt_length:
            self.biz_logger.error(f"参数校验失败：prompt 超长，长度={len(prompt)}")
            raise ValueError(f"prompt长度不能超过{self.max_prompt_length}")


    # ============ 校验组合 messages 的结构 ============
    def _validate_message_list(self, messages: list) -> None:
        """校验消息列表的结构、角色和顺序，非法则抛 ValueError。"""
        if len(messages) == 0:
            self.biz_logger.error("参数校验失败：messages 为空列表")
            raise ValueError("messages不能为空列表")
        for index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                self.biz_logger.error(f"参数校验失败：第{index}条message不是字典")
                raise ValueError(f"第{index}条message格式不正确")
            if "role" not in message or message.get("role") is None:
                self.biz_logger.error(f"参数校验失败：第{index}条message缺少role")
                raise ValueError(f"第{index}条message缺少role")
            if "content" not in message or message.get("content") is None:
                self.biz_logger.error(f"参数校验失败：第{index}条message缺少content")
                raise ValueError(f"第{index}条message缺少content")
            if set(message) != {"role", "content"}:
                self.biz_logger.error(f"参数校验失败：第{index}条message含有role、content以外的字段")
                raise ValueError(f"第{index}条message只能包含role和content")
            expected_role = "user" if index % 2 == 1 else "assistant"
            if message["role"] != expected_role:
                self.biz_logger.error(f"参数校验失败：第{index}条message的role必须为{expected_role}")
                raise ValueError(f"第{index}条message的role必须为{expected_role}")
            if not isinstance(message["content"], str):
                self.biz_logger.error(f"参数校验失败：第{index}条message的content不是字符串")
                raise ValueError(f"第{index}条message的content必须为字符串")
        if len(messages) % 2 == 0:
            self.biz_logger.error("参数校验失败：messages 条数不是单数")
            raise ValueError("messages条数必须为1、3、5……这种单数")


    def _use_prompt(self, prompt) -> None:
        """单轮：prompt 必须是字符串。None 和其他类型都按不是字符串处理。"""
        if not isinstance(prompt, str):
            self.biz_logger.error("参数校验失败：prompt 不是字符串")
            raise ValueError("prompt必须为字符串")
        self._validate_prompt(prompt)
        self.biz_logger.info(f"收到prompt，长度={len(prompt)}")
        self.messages = [{"role": "user", "content": prompt}]


    def _use_messages(self, messages) -> None:
        """组合对话：messages 为消息列表，合法条数是 1、3、5……。"""
        if not isinstance(messages, list):
            self.biz_logger.error("参数校验失败：messages 不是列表")
            raise ValueError("messages必须为消息列表")
        self._validate_message_list(messages)
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                self._validate_prompt(content)
                self.biz_logger.info(f"收到prompt，长度={len(content)}")
        self.messages = messages


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
    def chat(self, prompt=None, messages=None, timeout=None, temperature=None, max_tokens=None, model=None):
        """
        【调用 DeepSeek，返回模型回复】
            :param prompt: 单轮字符串
            :param messages: 组合消息列表，条数为 1、3、5……
            :return: 模型返回的文本内容
        """
        if prompt is not None and messages is not None:
            self.biz_logger.error("参数校验失败：prompt 与 messages 同时传入")
            raise ValueError("prompt和messages只能传一个")
        if prompt is None and messages is None:
            self.biz_logger.error("参数校验失败：prompt 和 messages 必须传一个")
            raise ValueError("prompt和messages必须传一个")
        if messages is not None:
            self._use_messages(messages)
        else:
            self._use_prompt(prompt)
        resp = self.client.chat.completions.create(
            model=self.model if model is None else model,
            messages=self.messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            timeout=self.timeout if timeout is None else timeout
        )

        # API返回结果校验。只看第一条 choice，取不到或对不上就说明原因，不重试
        try:
            choices = resp.choices
        except AttributeError:
            self.biz_logger.error("返回结果没有choices属性")
            raise RuntimeError("返回结果没有choices属性")
        if not isinstance(choices, list):
            self.biz_logger.error("choices必须为列表")
            raise RuntimeError("choices必须为列表")
        if len(choices) == 0:
            self.biz_logger.error("API 返回空choices")
            raise RuntimeError("API 返回空choices")

        try:
            message = choices[0].message
        except AttributeError:
            self.biz_logger.error("响应缺少message")
            raise RuntimeError("响应缺少message")

        try:
            content = message.content
        except AttributeError:
            self.biz_logger.error("响应缺少content")
            raise RuntimeError("响应缺少content")

        if not isinstance(content, str) or len(content.strip()) == 0:
            self.biz_logger.error("content必须为去空格后非空的字符串")
            raise RuntimeError("content必须为去空格后非空的字符串")

        self.biz_logger.info("API 调用成功")
        return content
    
