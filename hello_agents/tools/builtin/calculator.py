"""计算器工具（第一个内置工具，也是 ReAct 最经典的练手工具）

实现思路：不用 eval（不安全），而是用 Python 的 ast 模块把表达式解析成“语法树”，
再自己递归计算。这样既能算 2+3*4、sqrt(16)、sin(pi/2)，又不会执行任意危险代码。
"""

import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool, ToolParameter


class CalculatorTool(Tool):
    """安全的 Python 表达式计算器。"""

    # 支持二元运算符（加/减/乘/除/幂/异或）
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    # 支持的函数和常量
    FUNCTIONS = {
        "abs": abs, "round": round, "max": max, "min": min, "sum": sum,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "exp": math.exp, "pi": math.pi, "e": math.e,
    }

    def __init__(self):
        super().__init__(
            name="python_calculator",
            description="执行数学计算。支持基本运算和数学函数，例如：2+3*4、sqrt(16)、sin(pi/2)。",
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        # 兼容两种参数名：input 或 expression
        expr = parameters.get("input", "") or parameters.get("expression", "")
        if not expr:
            return "错误：计算表达式不能为空"
        try:
            node = ast.parse(expr, mode="eval")      # 解析成表达式语法树
            result = self._eval_node(node.body)       # 递归计算
            return str(result)
        except Exception as e:
            return f"计算失败: {e}"

    def _eval_node(self, node):
        """递归遍历语法树并求值。"""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left), self._eval_node(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            func = self.FUNCTIONS.get(node.func.id)
            if func is None:
                raise ValueError(f"不支持的函数: {node.func.id}")
            args = [self._eval_node(a) for a in node.args]
            return func(*args)
        if isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            raise ValueError(f"未定义的变量: {node.id}")
        raise ValueError(f"不支持的表达式类型: {type(node)}")

    def get_parameters(self) -> list:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="要计算的数学表达式，支持基本运算和数学函数",
                required=True,
            )
        ]


def calculate(expression: str) -> str:
    """便捷函数：直接算一个表达式。"""
    return CalculatorTool().run({"input": expression})
