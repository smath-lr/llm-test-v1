# LLM API 自动化测试练习

使用 Pytest + DeepSeek API 实现 LLM 输出的自动化验证

## 技术栈
- Python 3.10
- pytest
- openai SDK
- PyYAML

## 项目结构
\`\`\`
llm-test-v1
|—— config/         # 配置文件 （API Key 不提交）
|—— utils/          # 公共工具函数
|—— tests/          # pytest 测试用例
|—— requirements.txt
\`\`\`

## 运行方法
\`\`\`bash
pip install -r requirements.txt
pytest tests/ -v
\`\`\`