# LLM API 客户端 chat() 接口测试用例

> 被测对象：Python 封装的 LLM API 客户端 `chat()` 接口（基于 OpenAI SDK 兼容接口调用 DeepSeek API）
> 测试方式：Mock 单元测试为主，辅以少量真实 API 冒烟测试
> 用例说明：每条用例的「预期结果」描述业务预期行为，「断言」描述代码层面的校验点

## 模块一：Prompt / messages 输入校验

| 模块 | 用例 ID | 用例名称 | 优先级 | 前置条件 | 测试步骤 | 预期结果 | 断言 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Prompt / messages 输入校验 | TC-001 | 单轮 prompt 为空字符串 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="")` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt不能为空」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-002 | prompt 与 messages 都未传 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt=None, messages=None)` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt和messages必须传一个」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-003 | prompt 与 messages 同时传入 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="你好", messages=[{"role":"user","content":"你好"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt和messages只能传一个」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-004 | prompt 不是字符串 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt=1)` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt必须为字符串」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-005 | prompt 为空字符串且未传 messages | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="", messages=None)` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt不能为空」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-006 | prompt 为仅空格 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="  ")` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt不能为空」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-007 | messages 不是列表 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages={})` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「messages必须为消息列表」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-008 | messages 为空列表 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「messages不能为空列表」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-009 | 第 1 条 message 不是字典 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=["你好"])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message格式不正确」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-010 | 第 1 条 message 缺少 role 字段 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"content":"你好"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message缺少role」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-011 | 第 1 条 message 的 role 为 None | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":None,"content":"你好"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message缺少role」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-012 | 第 1 条 message 缺少 content 字段 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message缺少content」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-013 | 第 1 条 message 的 content 为 None | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":None}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message缺少content」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-014 | 第 1 条 message 含多余字段 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":"你好","name":"小明"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message只能包含role和content」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-015 | 第 1 条 message 的 role 为 assistant | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"assistant","content":"你好"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message的role必须为user」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-016 | 第 1 条 message 的 role 为 system | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"system","content":"你好"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message的role必须为user」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-017 | 第 2 条 message 的 role 为 user | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":"你好"},{"role":"user","content":"在吗"},{"role":"user","content":"继续"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第2条message的role必须为assistant」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-018 | 第 2 条 message 的 role 为 system | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":"你好"},{"role":"system","content":"在吗"},{"role":"user","content":"继续"}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第2条message的role必须为assistant」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-019 | 单条 user 的 content 为空字符串 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":""}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt不能为空」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-020 | 单条 user 的 content 为仅空格 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":"  "}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「prompt不能为空」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-021 | 单条 user 的 content 不是字符串 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":1}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第1条message的content必须为字符串」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-022 | 第 2 条 message 的 content 为整数 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 传入三条 messages，第 2 条 content 为整数，调用 `chat(messages=...)` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第2条message的content必须为字符串」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-023 | 第 2 条 message 的 content 为列表 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 传入三条 messages，第 2 条 content 为 `["在"]`，调用 `chat(messages=...)` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「第2条message的content必须为字符串」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-024 | messages 条数为偶数 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":"你好"},{"role":"assistant","content":"在吗"}])` | 角色顺序正确，但条数不是 1、3、5……，触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案为「messages条数必须为1、3、5……这种单数」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-025 | 单轮 prompt 长度为 1 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="A")` | 输入校验通过，正常发起模拟函数调用 | 返回 mock 文本「长度校验通过」，mock 函数调用 1 次 |
| Prompt / messages 输入校验 | TC-026 | 单轮 prompt 长度为 2 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="AA")` | 输入校验通过，正常发起模拟函数调用 | 返回 mock 文本「长度校验通过」，mock 函数调用 1 次 |
| Prompt / messages 输入校验 | TC-027 | 单轮 prompt 长度为 199 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat()`，prompt 为 199 个字符 | 输入校验通过，正常发起模拟函数调用 | 返回 mock 文本「长度校验通过」，mock 函数调用 1 次 |
| Prompt / messages 输入校验 | TC-028 | 单轮 prompt 长度为 200 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat()`，prompt 为 200 个字符 | 输入校验通过，达到上限，正常发起模拟函数调用 | 返回 mock 文本「长度校验通过」，mock 函数调用 1 次 |
| Prompt / messages 输入校验 | TC-029 | 单轮 prompt 长度为 201 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat()`，prompt 为 201 个字符 | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案匹配「prompt长度不能超过」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-030 | 一条 messages 的 user 内容长度为 201 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(messages=[{"role":"user","content":201个字符}])` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案匹配「prompt长度不能超过」，mock 函数调用 0 次 |
| Prompt / messages 输入校验 | TC-031 | 三条 messages 的最后一条 user 内容长度为 201 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 传入 user、assistant、user 三条消息，最后一条 user 内容为 201 个字符，调用 `chat(messages=...)` | 触发输入校验异常，不发起模拟函数调用 | 捕获 ValueError，文案匹配「prompt长度不能超过」，mock 函数调用 0 次 |

## 模块二：请求组装与参数透传

| 模块 | 用例 ID | 用例名称 | 优先级 | 前置条件 | 测试步骤 | 预期结果 | 断言 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 请求组装与参数透传 | TC-032 | 单轮对话 messages 组装 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="简单介绍自动化测试")` | 请求中 messages 为一条 user 消息，content 等于原 prompt，正常发起请求 | 返回 mock 文本「单轮参数传递成功」，mock 函数调用 1 次；messages 仅 1 条，且只含 role、content，role 为 user，content 等于原 prompt |
| 请求组装与参数透传 | TC-033 | 多轮对话 messages 原样传递 | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 传入 user、assistant、user 三条消息，调用 `chat(messages=...)` | 请求中的角色、顺序、内容与入参一致，正常发起请求 | 返回 mock 文本「多轮参数传递成功」，mock 函数调用 1 次；每条只含 role、content，角色、顺序、内容与入参一致 |
| 请求组装与参数透传 | TC-034 | 默认请求参数与假 api_key | P0 | Mock 模拟 DeepSeek 正常响应 | 1. 不改写参数，调用 `chat(prompt="校验默认请求参数")` | model、temperature、max_tokens、timeout、base_url 使用配置值，api_key 使用假密钥 | 返回 mock 文本「默认参数传递成功」，mock 函数调用 1 次；上述五项等于配置，api_key 等于假密钥且不等于环境变量中的真实密钥。不断言 messages |
| 请求组装与参数透传 | TC-035 | 只改写 model | P1 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(prompt="校验改写后的请求参数", model="mock-model")` | 请求使用新 model，其余请求参数仍用配置 | 返回 mock 文本「改写参数传递成功」，mock 函数调用 1 次；model 等于新值且不等于配置，temperature、max_tokens、timeout 仍等于配置；客户端实例上的这四项保持配置值 |
| 请求组装与参数透传 | TC-036 | 只改写 temperature | P1 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(..., temperature=0.2)` | 请求使用新 temperature，其余请求参数仍用配置 | 返回 mock 文本「改写参数传递成功」，mock 函数调用 1 次；temperature 等于 0.2 且不等于配置，其余三项仍等于配置；客户端实例上的这四项保持配置值 |
| 请求组装与参数透传 | TC-037 | 只改写 max_tokens | P1 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(..., max_tokens=128)` | 请求使用新 max_tokens，其余请求参数仍用配置 | 返回 mock 文本「改写参数传递成功」，mock 函数调用 1 次；max_tokens 等于 128 且不等于配置，其余三项仍等于配置；客户端实例上的这四项保持配置值 |
| 请求组装与参数透传 | TC-038 | 只改写 timeout | P1 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(..., timeout=8)` | 请求使用新 timeout，其余请求参数仍用配置 | 返回 mock 文本「改写参数传递成功」，mock 函数调用 1 次；timeout 等于 8 且不等于配置，其余三项仍等于配置；客户端实例上的这四项保持配置值 |
| 请求组装与参数透传 | TC-039 | 同时改写 model 和 temperature | P1 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(..., model="mock-model", temperature=0.2)` | 请求使用新 model 和 temperature，max_tokens、timeout 仍用配置 | 返回 mock 文本「改写参数传递成功」，mock 函数调用 1 次；传入的两项等于新值且不等于配置，未传入的两项仍等于配置；客户端实例上的这四项保持配置值 |
| 请求组装与参数透传 | TC-040 | 同时改写 model、temperature、max_tokens、timeout | P1 | Mock 模拟 DeepSeek 正常响应 | 1. 调用 `chat(..., model="mock-model", temperature=0.2, max_tokens=128, timeout=8)` | 四项请求参数都使用新值 | 返回 mock 文本「改写参数传递成功」，mock 函数调用 1 次；四项都等于新值且不等于配置；客户端实例上的这四项保持配置值 |

## 模块三：重试与异常请求

| 模块 | 用例 ID | 用例名称 | 优先级 | 前置条件 | 测试步骤 | 预期结果 | 断言 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 重试与异常请求 | TC-041 | 400 参数错误不重试 | P0 | Mock 模拟 HTTP 400，异常为 BadRequestError | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的参数错误 | 捕获 BadRequestError，状态码 400，mock 函数调用 1 次 |
| 重试与异常请求 | TC-054 | 401 认证失败不重试 | P0 | Mock 模拟 HTTP 401，异常为 AuthenticationError | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的认证错误 | 捕获 AuthenticationError，状态码 401，mock 函数调用 1 次 |
| 重试与异常请求 | TC-055 | 403 无权限不重试 | P0 | Mock 模拟 HTTP 403，异常为 PermissionDeniedError | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的权限错误 | 捕获 PermissionDeniedError，状态码 403，mock 函数调用 1 次 |
| 重试与异常请求 | TC-056 | 404 资源不存在不重试 | P0 | Mock 模拟 HTTP 404，异常为 NotFoundError | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的未找到错误 | 捕获 NotFoundError，状态码 404，mock 函数调用 1 次 |
| 重试与异常请求 | TC-042 | 429 没有 Retry-After 不重试 | P0 | Mock 模拟 HTTP 429 的 RateLimitError，响应头没有 Retry-After | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的限流错误 | 捕获 RateLimitError，状态码 429，mock 函数调用 1 次 |
| 重试与异常请求 | TC-057 | 429 的 Retry-After 为空白不重试 | P0 | Mock 模拟 RateLimitError，Retry-After 为空白 | 1. 使用合法 prompt 调用 `chat()` | 空白头视为没有，只请求 1 次 | 捕获 RateLimitError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-058 | 429 的 Retry-After 无法解析不重试 | P0 | Mock 模拟 RateLimitError，Retry-After 为非数字 | 1. 使用合法 prompt 调用 `chat()` | 解析失败视为没有，只请求 1 次 | 捕获 RateLimitError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-059 | 429 的 Retry-After 为负数不重试 | P0 | Mock 模拟 RateLimitError，Retry-After 为负数 | 1. 使用合法 prompt 调用 `chat()` | 负数视为没有，只请求 1 次 | 捕获 RateLimitError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-060 | 429 的 Retry-After 为过去的日期不重试 | P0 | Mock 模拟 RateLimitError，Retry-After 为过去的 HTTP 日期 | 1. 使用合法 prompt 调用 `chat()` | 解析出的秒数为负，视为没有，只请求 1 次 | 捕获 RateLimitError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-061 | 429 但不是 RateLimitError 不重试 | P0 | Mock 模拟状态码 429，异常不是 RateLimitError，没有 Retry-After | 1. 使用合法 prompt 调用 `chat()` | 没有可用的 Retry-After，只请求 1 次 | 捕获原来的异常，状态码 429，且不是 RateLimitError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-062 | 500 的 body 不是字典不重试 | P0 | Mock 模拟 HTTP 500 的 InternalServerError，body 不是字典 | 1. 使用合法 prompt 调用 `chat()` | 无法识别临时故障，只请求 1 次 | 捕获 InternalServerError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-063 | 500 的 type 和 code 都未命中不重试 | P0 | Mock 模拟 HTTP 500 的 InternalServerError，body 为字典，type 和 code 都是临时故障集合外的字符串 | 1. 使用合法 prompt 调用 `chat()` | 未命中临时故障，只请求 1 次 | 捕获 InternalServerError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-043 | 500 的 code 为整数不重试 | P0 | Mock 模拟 HTTP 500，body 的 code 为整数 500 | 1. 使用合法 prompt 调用 `chat()` | 未识别为临时故障，只请求 1 次 | 捕获 InternalServerError，状态码 500，mock 函数调用 1 次 |
| 重试与异常请求 | TC-044 | 500 只在 message 里出现临时故障文案不重试 | P0 | Mock 模拟 HTTP 500，message 为 server_error，type 和 code 都不是临时故障字符串 | 1. 使用合法 prompt 调用 `chat()` | 不看 message 文案，只请求 1 次 | 捕获 InternalServerError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-064 | 500 但类型不是 InternalServerError 不重试 | P0 | Mock 模拟状态码 500，异常不是 InternalServerError，body 含 server_error | 1. 使用合法 prompt 调用 `chat()` | 类型不符，不看 body，只请求 1 次 | 捕获原来的异常，状态码 500，且不是 InternalServerError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-065 | APITimeoutError 没有 cause 不重试 | P0 | Mock 模拟 APITimeoutError，没有 cause | 1. 使用合法 prompt 调用 `chat()` | 超时一律不重试，只请求 1 次 | 捕获 APITimeoutError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-047 | APITimeoutError 的 cause 为 ConnectTimeout 不重试 | P0 | Mock 模拟 APITimeoutError，cause 为 ConnectTimeout | 1. 使用合法 prompt 调用 `chat()` | 超时一律不重试，只请求 1 次 | 捕获 APITimeoutError，cause 为 ConnectTimeout，mock 函数调用 1 次 |
| 重试与异常请求 | TC-045 | APITimeoutError 的 cause 为 ReadTimeout 不重试 | P0 | Mock 模拟 APITimeoutError，cause 为 ReadTimeout | 1. 使用合法 prompt 调用 `chat()` | 超时一律不重试，只请求 1 次 | 捕获 APITimeoutError，cause 为 ReadTimeout，mock 函数调用 1 次 |
| 重试与异常请求 | TC-066 | APITimeoutError 的 cause 为 ConnectError 不重试 | P0 | Mock 模拟 APITimeoutError，cause 为 ConnectError | 1. 使用合法 prompt 调用 `chat()` | 超时一律不重试，不看来源，只请求 1 次 | 捕获 APITimeoutError，cause 为 ConnectError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-067 | APITimeoutError 的 cause 为 RemoteProtocolError 不重试 | P0 | Mock 模拟 APITimeoutError，cause 为 RemoteProtocolError | 1. 使用合法 prompt 调用 `chat()` | 超时一律不重试，不看来源，只请求 1 次 | 捕获 APITimeoutError，cause 为 RemoteProtocolError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-068 | APIConnectionError 没有 cause 不重试 | P0 | Mock 模拟 APIConnectionError，没有 cause | 1. 使用合法 prompt 调用 `chat()` | 没有可重试的 cause，只请求 1 次 | 捕获 APIConnectionError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-069 | APIConnectionError 的 cause 为 ConnectTimeout 不重试 | P0 | Mock 模拟 APIConnectionError，cause 为 ConnectTimeout | 1. 使用合法 prompt 调用 `chat()` | ConnectTimeout 不是 ConnectError，只请求 1 次 | 捕获 APIConnectionError，cause 为 ConnectTimeout，mock 函数调用 1 次 |
| 重试与异常请求 | TC-046 | DNS 失败不重试 | P0 | Mock 模拟 APIConnectionError，cause 为 ConnectError，文本含 nodename nor servname provided | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的连接错误 | 捕获 APIConnectionError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-070 | 另一种 DNS 文案不重试 | P0 | Mock 模拟 APIConnectionError，cause 为 ConnectError，文本含 name or service not known | 1. 使用合法 prompt 调用 `chat()` | 只请求 1 次，抛出原来的连接错误 | 捕获 APIConnectionError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-048 | 500 临时故障连续失败 3 次 | P0 | Mock 模拟 HTTP 500，body 的 type 为 server_error；抖动固定为 0 | 1. 使用合法 prompt 调用 `chat()` | 重试 2 次后仍失败，抛出原来的服务端错误 | 捕获 InternalServerError，mock 函数调用 3 次；客户端的 model、temperature、max_tokens、timeout 仍为配置值 |
| 重试与异常请求 | TC-049 | 429 的 Retry-After 超过预算不重试 | P0 | Mock 模拟 HTTP 429，Retry-After 为 30 秒 | 1. 使用合法 prompt 调用 `chat()` | 剩余时间盖不住 30 秒再加 2 秒，睡眠前抛出原来的限流错误 | 捕获 RateLimitError，mock 函数调用 1 次 |
| 重试与异常请求 | TC-050 | 500 临时故障失败 1 次后成功 | P0 | Mock 先返回 type 为 server_error 的 500，再返回正常文本；抖动固定为 0 | 1. 使用合法 prompt 调用 `chat()` | 第 2 次返回模拟文本 | 返回 mock 文本「重试后成功」，mock 函数调用 2 次；客户端的 model、temperature、max_tokens、timeout 仍为配置值 |
| 重试与异常请求 | TC-051 | 500 临时故障失败 2 次后成功 | P0 | Mock 连续两次返回 type 为 server_error 的 500，第 3 次返回正常文本；抖动固定为 0 | 1. 使用合法 prompt 调用 `chat()` | 第 3 次返回模拟文本 | 返回 mock 文本「重试后成功」，mock 函数调用 3 次 |
| 重试与异常请求 | TC-052 | 429 带 Retry-After 0 失败 1 次后成功 | P0 | Mock 先返回带 Retry-After 0 的 429，再返回正常文本 | 1. 使用合法 prompt 调用 `chat()` | 按头里的 0 秒等待后，第 2 次返回模拟文本 | 返回 mock 文本「重试后成功」，mock 函数调用 2 次 |
| 重试与异常请求 | TC-053 | 429 带 Retry-After 0 失败 2 次后成功 | P0 | Mock 连续两次返回带 Retry-After 0 的 429，第 3 次返回正常文本 | 1. 使用合法 prompt 调用 `chat()` | 第 3 次返回模拟文本 | 返回 mock 文本「重试后成功」，mock 函数调用 3 次 |

## 模块四：服务端响应解析

| 模块 | 用例 ID | 用例名称 | 优先级 | 前置条件 | 测试步骤 | 预期结果 | 断言 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 服务端响应解析 | TC-033 | 正常响应解析提取文本 | P1 | Mock 返回 DeepSeek 正常 JSON 格式响应报文 | 1. 调用 `chat()` 执行请求 | 成功完整提取返回的 JSON 格式值 | 正常返回 mock 模拟的返回文本，键值对一一对应 |
| 服务端响应解析 | TC-034 | 响应 choices 为空列表解析处理 | P1 | Mock 返回 choices 为空列表的响应报文 | 1. 调用 `chat()` 执行请求 | 抛出 RuntimeError 解析异常，不重试 | 捕获 RuntimeError 异常，mock 函数仅调用 1 次 |
| 服务端响应解析 | TC-035 | 响应 choices 残缺结构解析处理 | P1 | Mock 返回 choices 结构残缺的异常响应 | 1. 调用 `chat()` 执行请求 | 抛出 AttributeError 解析异常，不重试 | 捕获 AttributeError 异常，mock 函数仅调用 1 次 |

## 模块五：真实 API 冒烟测试

| 模块 | 用例 ID | 用例名称 | 优先级 | 前置条件 | 测试步骤 | 预期结果 | 断言 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 真实 API 冒烟测试 | TC-036 | 真实 DeepSeek 接口单轮对话正常常识提问 | P0 | 具备测试专用 API 密钥，网络可连通 DeepSeek 服务，尝试提问【100*3 等于几】 | 1. 真实环境调用 `chat()` 传入简短合法 prompt | 客户端无崩溃，返回非空结果，且结果包含关键字词 | 返回结果非空，且包含关键字词【300 或三百】 |
| 真实 API 冒烟测试 | TC-037 | 真实 DeepSeek 接口多轮对话正常常识提问 | P0 | 具备测试专用 API 密钥，网络可连通 DeepSeek 服务，多轮提问【小明买了 5 个桃子吃了 1 个，还剩几个】、【又吃了 2 个还剩几个】 | 1. 真实环境调用 `chat()` 传入简短合法 prompt | 客户端无崩溃，返回非空结果，且结果包含关键字词 | 返回结果非空，且包含关键字词【2 或二】 |
| 真实 API 冒烟测试 | TC-038 | 真实 DeepSeek 接口 prompt 有害提问，模型拒绝回答 | P1 | 具备测试专用 API 密钥，网络可连通 DeepSeek 服务 | 1. 真实环境调用 `chat()` 传入有害提问 prompt；2. 共执行 2 次对话 | 客户端无崩溃，返回结果包含拒绝回答的关键字词 | 每次返回结果非空，且包含【抱歉、不能、无法、对不起、拒绝】中任一 |
| 真实 API 冒烟测试 | TC-039 | 真实 DeepSeek 接口传入超短请求响应时长返回异常 | P1 | 具备测试专用 API 密钥，网络可连通 DeepSeek 服务 | 1. 真实环境调用 `chat()` 传入合法 prompt 及 0.05s 的 timeout | 连续失败 3 次，重试耗尽后向外抛出超时异常，请求共调用 3 次 | 捕获 APITimeoutError 异常，API 接口共调用 3 次 |


