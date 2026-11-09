# LLM API 客户端测试设计

## 1. 被测对象

自己封装的 Python LLM API 客户端，主要入口为 `chat()`，通过 OpenAI SDK 兼容接口调用 DeepSeek API。

测试目的：验证调用能力、输入处理、异常处理及响应解析是否符合预期。

## 2. 测试规则

1. **输入处理**：Prompt 内容非空、长度符合 1～200 个字符的限制，且能够正确转换为 API 所需的 `messages` 格式。
2. **请求调用**：单轮和多轮对话请求参数传递符合预期，包括 `api_key`、`base_url`、`timeout`、`model`、`temperature`、`max_tokens` 等。
3. **异常处理**：覆盖限流、服务端临时故障和网关状态、连接失败、超时、其他客户端错误、未识别的服务端错误、调用前的参数错误、响应解析错误，以及加载配置时的密钥和配置文件错误。可重试的场景要确认会按最多 3 次、剩余预算大于等待加 2 秒继续请求；不可重试的场景要确认原样抛出、不再请求。
4. **响应解析**：正常场景能够正确提取响应文本。没有 `choices`、`choices` 不是列表、空 `choices`、缺少 `message` 或 `content`，以及 `content` 不是去空格后的非空字符串时，抛出 `RuntimeError` 且不重试。
5. **单轮和多轮**：`prompt` 与 `messages` 的角色、顺序和内容按预期传递。
6. **流式输出**：未来扩展，不纳入当前验收目标。

## 3. 测试策略

### 执行方式

1. **Mock 测试**：验证输入处理、请求参数传递、异常处理、重试规则及响应解析。
2. **真实 API 冒烟测试**：当前四条。配置中的 `api_key` 为 `***`，请求头携带 `.env` 原文；单轮计算（`23乘45等于多少`，回答含「1035」）；多轮翻译（`good morning`，回答含「早上好」）；密钥缺失或空字符串在加载配置时拦住。不测有害提问。

### 设计技术

1. **等价类、边界值**：验证 Prompt 和请求参数的有效、无效输入及边界情况。
2. **异常测试**：验证请求超时、连接失败、参数错误等异常场景。
3. **场景测试**：验证多轮对话、重试后恢复、重试耗尽、预算不足停止等完整流程。

## 4. 环境与固定值

- 操作系统：Windows
- Python：3.10.22
- pytest：9.1.1
- openai：3.28.0
- API 服务：DeepSeek API

| 配置项 | 配置内容 |
| --- | --- |
| API Key | 环境变量 `DEEPSEEK_API_KEY` 保存原文，并用于真实请求。`load_llm_config()` 返回的 `api_key` 是打码值 `***` |
| Base URL | `https://api.deepseek.com`，来自 `config/config.yaml` |
| Model | `deepseek-flash`，来自 `config/config.yaml` |
| Timeout | 15 秒，来自 `config/config.yaml`。这是单次 HTTP 超时 |
| Temperature | 0.7，来自 `config/config.yaml` |
| Max tokens | 500，来自 `config/config.yaml` |
| Prompt 上限 | 200 个字符，来自 `src/llm_client.py` 的 `MAX_PROMPT_LENGTH` |

重试固定值来自 `src/llm_client.py`：

- 最多重试 2 次（`MAX_RETRIES = 2`）。加上首次请求，最多尝试 3 次（`MAX_ATTEMPTS = 3`）。
- 一次 `chat()` 的重试总预算是 25 秒（`TOTAL_BUDGET_SECONDS = 25`）。剩余时间大于本次等待再加上 2 秒（`ATTEMPT_SECONDS = 2`）时，发起下一次请求。

## 5. 测试数据

1. **正常 Prompt**：有效的普通文本，验证能够正常发送请求并提取回复。
2. **空 Prompt**：空字符串和仅空格按照输入校验规则拒绝，抛出 ValueError「prompt不能为空」，不发起请求。
3. **超长 Prompt**：覆盖正常长度、接近上限、达到 200、超过 200。1～200 个字符通过，超过 200 拒绝。
4. **多轮对话数据**：包含不同消息角色、顺序及内容，验证消息结构和内容是否正确传递。
5. **异常模拟数据**：模拟不同异常类型，验证对应的异常处理、重试次数及最终结果。

## 6. 重试与异常

依据 `src/llm_client.py` 的 `_retry_interval`、`_is_retryable_connection`，以及 `utils/config_loader.py`。SDK 的 `max_retries` 为 0，重试只由客户端自己决定。

可重试时，还要同时满足：这是第 1 或第 2 次失败（最多尝试 3 次），并且剩余预算大于本次等待加 2 秒。否则抛出这一次的原异常，不再请求。

### 可以重试

- **带 `Retry-After` 的 429 / `RateLimitError`**：响应头能解析成大于等于 0 的秒数，或未来的 HTTP 日期时才重试，等待时间用这个秒数。抛出的仍是原来的 `RateLimitError`。
- **临时故障的 HTTP 500**：类型必须是 `InternalServerError`，并且 body 里 `type` 或 `code` 是临时故障字符串（忽略大小写）。命中才重试，间隔走抖动。不看 `message`。
- **502、504**：重试，间隔走抖动，不使用 `Retry-After`。
- **503**：有可用的 `Retry-After` 就用它，否则走抖动。
- **非 DNS 的 `APIConnectionError`**：异常类型正好是 `APIConnectionError`（不是它的子类），并且 `__cause__` 是 `ConnectError` 或 `RemoteProtocolError`，文本里没有 DNS 标记 `nodename nor servname provided`、`name or service not known`。间隔走抖动。抛出的仍是原来的 `APIConnectionError`。

### 不能重试

- **超时 `APITimeoutError`**：不重试，原样抛出。它是 `APIConnectionError` 的子类，代码用精确类型判断，所以不算可重试的连接失败。
- **其他连接失败**：`APIConnectionError` 没有上述 cause、cause 是 `ConnectTimeout`，或文本含 DNS 标记。不重试，原样抛出。
- **429 没有可用的 `Retry-After`**：头缺失、空白、非数字、负数或过去的日期。不重试，抛出原来的限流异常。
- **其他 4xx**（如 400、401、403、404）：不重试，抛出原来的异常。
- **未识别为临时故障的 500**：类型不是 `InternalServerError`，body 不是字典，`type` 和 `code` 都未命中，`code` 是整数，或只有 `message` 里出现故障文案。不重试，抛出原来的异常。
- **输入校验 `ValueError`**：空 prompt、超长、`messages` 结构不合法等，在发请求前抛出。不重试，也不发请求。
- **响应解析 `RuntimeError`**：没有 `choices`、不是列表、空列表、缺少 `message` 或 `content`、`content` 不是去空格后的非空字符串。抛出对应的 `RuntimeError`，不重试。
- **密钥缺失或为空**：`load_llm_config()` 抛出 `ValueError`，文案为「环境变量 DEEPSEEK_API_KEY 未配置」。不创建客户端，不发请求，也不打码。
- **配置文件不存在**：抛出 `FileNotFoundError`，不创建客户端。

### 密钥打码

只有读到非空密钥时，`load_llm_config()` 才返回打码值 `***`。创建真实客户端时另行读取原文。缺失或空字符串走上面的配置错误，不打码。

## 7. 验收检查点

P0 须全部通过：

- 正常请求能够成功调用并返回文本。
- 正常 Prompt 能够正确转换为 messages 并传递。
- 只改写 `model` 时，请求使用新值，其余参数仍为配置值，客户端实例保持配置。
- 无效 Prompt 能够按照校验规则抛出异常，并停止 API 请求。
- 多轮对话 messages 的角色、顺序和内容能够正确传递。
- 请求超时不重试。可重试的连接异常按重试规则执行。
- 不需要重试的异常不会发生重试。
- 重试耗尽后能够正确抛出异常。
- 没有 `choices`、`choices` 不是列表、空 `choices`、缺少 `message` 或 `content`、`content` 不是字符串或为空字符串时，抛出 `RuntimeError` 且不重试。
- 配置中的 `api_key` 为 `***`，请求头携带 `.env` 中的原文。断言不输出密钥。
- 真实单轮调用的回答包含「1035」。
- 真实多轮调用的回答包含「早上好」。
- 密钥缺失或为空时，加载配置抛出「环境变量 DEEPSEEK_API_KEY 未配置」，不创建客户端、不发请求。
- 预算不足时停止重试，并抛出当次异常。
