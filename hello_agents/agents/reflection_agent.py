"""Reflection Agent（7.4 第三个范式：先答、再反思、再改进）

思路：不是一次给答案，而是“写初稿 → 当评委挑刺 → 改稿 → 再挑刺 → 再改”，
迭代 max_iterations 轮。适合写作、代码生成、方案设计这类“越改越好”的任务。

提示词工程味很浓：它靠三套提示词驱动——
    initial : 写初稿
    reflect : 当评审，挑毛病（或说“无需改进”）
    refine  : 根据反馈改稿
"""

from typing import Optional, List, Dict, Any

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，找出问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析回答质量，指出不足并提出具体改进建议。如果回答已经很好，请只回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
""",
}


class Memory:
    """短期记忆：记录每一轮的“尝试稿”和“反思反馈”，供下一轮参考。"""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆新增一条 '{record_type}' 记录。")

    def get_last_execution(self) -> str:
        """取最近一次“尝试稿”（execution 类型）的内容。"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""


class ReflectionAgent(Agent):
    """自我反思、迭代优化的智能体。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memory()
        # 用户自定义优先，否则用默认三套提示词
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """统一的 LLM 调用入口：一条 user 消息，拿完整回复。"""
        return self.llm.invoke([{"role": "user", "content": prompt}], **kwargs) or ""

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        self.memory = Memory()

        # 1) 初稿
        print("\n--- 初始尝试 ---")
        initial_result = self._get_llm_response(
            self.prompts["initial"].format(task=input_text), **kwargs
        )
        self.memory.add_record("execution", initial_result)

        # 2) 迭代：反思 → 判断 → 改进
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            last_result = self.memory.get_last_execution()
            feedback = self._get_llm_response(
                self.prompts["reflect"].format(task=input_text, content=last_result), **kwargs
            )
            self.memory.add_record("reflection", feedback)

            # 评审说“不用改”就提前收工，省 token
            if "无需改进" in feedback:
                print("\n✅ 反思认为已无需改进，任务完成。")
                break

            refined = self._get_llm_response(
                self.prompts["refine"].format(
                    task=input_text, last_attempt=last_result, feedback=feedback
                ), **kwargs
            )
            self.memory.add_record("execution", refined)

        final_result = self.memory.get_last_execution()
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        return final_result
