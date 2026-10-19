# LLM API 自动化测试练习
基于pytest实现大模型API自动化测试项目，用于验证LLM基础能力、边界输入、安全能力。

## 技术栈
- Python 3.10
- pytest
- openai SDK
- PyYAML

## 项目结构
```bash
llm-test-v1
├── config/                  # 配置目录，存放 API 密钥
│   ├── config.example.yaml  # 配置模板（可提交 git，仅模板，不含真实密钥）
│   └── config.yaml          # 本地私有密钥文件，.gitignore 忽略，不上传 git
├── logs/                    # 自动生成，存放运行日志与失败报文，.gitignore 忽略
│   ├── fail_payload/        # 保存测试失败时的完整 payload json 报文
│   ├── llm_client.log       # LLM 业务调用日志
│   └── test_run.log         # pytest 测试执行日志
├── utils/                   # 公共工具函数
│   └── llm_client.py        # LLM 客户端封装
├── tests/                   # pytest 测试用例集合
│   ├── conftest.py          # pytest 全局钩子、日志、失败报文保存逻辑
│   ├── test_llm_basic.py            # 基础冒烟用例
│   ├── test_api_response.py          # API 返回结构校验
│   ├── test_prompt_boundary.py       # Prompt 边界值测试
│   ├── test_prompt_equivalence.py    # 等价类 prompt 测试
│   └── test_full_flow.py             # 完整业务流程测试
├── requirements.txt
└── .gitignore
```

## 环境安装
建议先创建并激活 Python 虚拟环境，再安装依赖
```bash
pip install -r requirements.txt
```
## 常规执行，只在失败时输出控制台信息
`-v`：verbose，展示每条用例执行结果
```bash
pytest tests/ -v
```
## 调试模式：终端实时打印日志，方便开发排查问题
`-s`：stdout，开启控制台标准输出
```bash
pytest tests/ -v -s
```
## 测试覆盖范围
1. **基础冒烟测试**：验证大模型正常返回、基础问答、安全内容拒绝
2. **API返回校验**：真实请求校验正常返回值类型；mock 模拟空 choices 场景、服务 500 异常场景，验证异常捕获逻辑，mock 用例无需联网、不消耗 token
3. **边界用例**：参数化校验 Prompt 长度边界，覆盖最短有效输入、上限临界值、超限场景；验证超长 prompt 会触发参数校验异常
4. **等价类测试**：采用参数化，划分有效、无效等价类。有效类包含普通中文、特殊字符 + emoji；无效类覆盖空串、None、纯空白字符串、超长文本，校验输入合法性判断逻辑。
5. **全流程端到端测试（test_full_flow.py）**
    - 单轮链路：配置加载、prompt预处理、API调用、返回断言，验证完整单次调用链路；
    - 多轮对话：连续多轮交互，验证LLM上下文记忆能力，检验模型能否记住前文对话信息。