"""框架统一异常类。

为什么要自定义异常？
- 内置 Exception 太泛，没法区分"是框架出的问题还是别的 bug"
- 统一异常名后，上层代码可以精确捕获框架错误，做兜底处理
"""


class HelloAgentsException(Exception):
    """HelloAgents 框架的基础异常类。

    LLM 调用失败、配置缺失、工具执行出错等框架内部问题，
    都抛这个异常，调用方 except HelloAgentsException 即可精准兜底。
    """

    pass
