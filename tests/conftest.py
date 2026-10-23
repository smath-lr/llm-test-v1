import traceback
import logging
import pytest
from pathlib import Path
import time
import json

# 开启：logging内部handler发生IO错误时抛出异常，不要静默吞掉
logging.raiseExceptions = True
# 记录已经打印模块标题的文件，模块级变量
_printed_modules = set()

# pathlib：项目根目录/日志目录/失败用例JSON目录
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)  #不存在自动创建logs文件夹
TEST_LOG_FILE = LOG_DIR / "test_run.log"
FAIL_PAYLOAD_DIR = LOG_DIR / "fail_payload"
FAIL_PAYLOAD_DIR.mkdir(exist_ok=True)

# 测试日志logger（和业务logger互相独立）
_test_logger = logging.getLogger("llm_test")
_test_logger.setLevel(logging.INFO)
# 每次初始化清空旧handler，适配常驻worker场景
_test_logger.handlers.clear()

log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
# 新建控制台输出handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
_test_logger.addHandler(stream_handler)
# 新建文件输出handler
file_handler = logging.FileHandler(TEST_LOG_FILE, encoding="utf-8")
file_handler.setFormatter(log_formatter)
_test_logger.addHandler(file_handler)


@pytest.fixture
def valid_prompt():
    """fixture：复用的合法prompt，所有用例可以直接调用"""
    return "简单介绍自动化测试"
@pytest.fixture
def max_length_prompt():
    """fixture：刚好达到上限长度prompt"""
    return "a"*200


## ============ 用例执行失败，保存完整json信息 ============
def save_fail_payload(case_name: str, payload_data: dict) -> str:
    """保存失败用例完整payload，直接传入整个payload字典"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_case_name = case_name[:50].replace("[", "_").replace("]", "_").replace("/", "_")
    filename = f"{safe_case_name}_{timestamp}.json"
    save_path = FAIL_PAYLOAD_DIR / filename

    save_data = {
        "case_name": case_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **payload_data
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return str(save_path)


# ============ 用例前置钩子：日志记录模块名、用例名 ============
def pytest_runtest_setup(item):
    module_name = item.module.__name__
    if module_name not in _printed_modules:
        # 直接拿已经创建好的logger实例
        _test_logger.info(f"\n===== 【模块开始执行】模块名称：{module_name} =====")
        _printed_modules.add(module_name)
    # 每次用例都打印当前用例名称
    _test_logger.info(f"【测试用例开始】用例名称：{item.name}")


# ============ 用例执行钩子：捕获用例结果，失败打印完整堆栈日志 ============
def pytest_runtest_makereport(item, call):
    # call.when 可选：setup / call / teardown
    # 只处理用例实际执行阶段
    if call.when != "call":
        return

    payload_data = item.stash.get("payload_data", None)
    if not payload_data:
        # 没有payload数据，直接跳过
        return

    prompt_len = payload_data["prompt_len"]
    resp_len = payload_data["response_len"]
    expected_exception = payload_data["expected_exception"]

    if call.excinfo is not None:
        # 用例失败，保存json文件 + 输出错误日志+堆栈
        case_full_name = item.name
        json_path = save_fail_payload(case_full_name, payload_data)
        _test_logger.error(f"【用例失败】用例名称：{case_full_name}，prompt长度：{prompt_len}，返回长度：{resp_len}")
        _test_logger.error(f"【用例失败】完整请求响应已保存至：{json_path}")
        tb_str = "".join(traceback.format_exception(call.excinfo.type, call.excinfo.value, call.excinfo.tb))
        _test_logger.error(f"异常堆栈信息：\n{tb_str}")
    else:
        # 用例成功：区分普通成功 / 捕获预期异常成功
        if not expected_exception:
            _test_logger.info(f"【用例执行成功】用例名称：{item.name}，prompt长度：{prompt_len}，返回长度：{resp_len}")
        else:
            _test_logger.info(f"【用例执行成功（捕获预期异常）】用例名称：{item.name}, prompt长度:{prompt_len}, 返回长度:{resp_len}, 捕获异常：{expected_exception}")


# ============ 会话结束钩子，测试全部跑完自动清理logger ============
def pytest_sessionfinish():
    # 会话全部执行完毕，清理logger handler
    for handler in _test_logger.handlers[:]:
        handler.close()
        _test_logger.removeHandler(handler)