"""ReAct Agent（7.4 第二个范式 + 7.5 工具系统的组合）

ReAct = Reasoning + Acting（推理 + 行动）。
它不是一问一答，而是一个“思考→行动→观察”的循环：

    Thought: 我需要先算一下 3.7 的平方...
    Action: python_calculator[3.7**2]
    Observation: 13.69
    Thought: 再乘以 12 加 88.4...
    Action: python_calculator[13.69*12+88.4]
    Observation: 252.68
    Thought: 已经有答案了
    Action: Finish[252.68]

关键设计（这也是官方框架踩过的坑，我直接修掉了）：
    - 提示词里用纯文本 "Thought:" / "Action:"（不加粗），
      和下面的解析正则 r"Action: (.*)" 严格对应，否则模型一加粗就解析失败。
    - ReActAgent 必须传 tool_registry，因为“列工具清单 + 执行工具”全靠它。
"""

import re
from typing import Optional, List, Tuple

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry


# 默认提示词模板：占位符 {tools} {question} {history} 由 run() 填充
DEFAULT_REACT_PROMPT = """你是一个具备推理与行动能力的AI助手。你可以分析问题、调用工具获取信息、再推理，直到能给出准确答案。

## 可用工具
{tools}

## 工作流程
每次只执行一个步骤，严格按以下格式回应：
Thought: 分析当前问题，思考下一步该做什么。
Action: 二选一：
- 调用工具：工具名[参数]
- 完成任务：Finish[最终答案]

## 当前任务
Question: {question}

## 执行历史
{history}

现在开始："""


class ReActAgent(Agent):
    """ReAct（推理+行动）智能体：通过 Thought/Action/Observation 循环解决问题。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,        # 必传！没有注册表就无法列工具、调工具
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,                 # 最多循环几轮，防止死循环烧钱
        custom_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []  # 记录 Action/Observation 的纯文本日志
        # 用户自定义优先，否则用默认模板（已修正格式，避免加粗导致解析失败）
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    def run(self, input_text: str, **kwargs) -> str:
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1) 拼提示词：工具清单 + 问题 + 历史
            prompt = self.prompt_template.format(
                tools=self.tool_registry.get_tools_description(),
                question=input_text,
                history="\n".join(self.current_history),
            )

            # 2) 调 LLM（注意：这里只发一条 user 消息，不像 SimpleAgent 拼历史）
            response_text = self.llm.invoke([{"role": "user", "content": prompt}], **kwargs)
            if not response_text:
                print("❌ LLM 未返回有效响应。")
                break

            # 3) 解析出 Thought 和 Action
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                print("⚠️ 未能解析出有效的 Action，流程终止。")
                break

            # 4) 如果模型认为可以收尾 → Finish
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            # 5) 否则解析“工具名[参数]”，执行工具
            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                self.current_history.append("Observation: 无效的 Action 格式，请检查。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")

            # 6) 把这一轮的 Action/Observation 写进历史，进入下一轮循环
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        print("⏰ 已达到最大步数，流程终止。")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    # ---------- 解析逻辑（提示词与正则必须成对设计）----------

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """从模型输出里提取 Thought 和 Action。"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析 '工具名[参数]' 形式。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """提取 Finish[答案] 里的答案部分。"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""
