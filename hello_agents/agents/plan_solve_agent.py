"""Plan-and-Solve Agent（7.4 第四个范式：先规划、再逐步执行）

思路：遇到复杂任务，先让 LLM 把问题拆成一串步骤（计划），
再一个步骤一个步骤地执行，每步的结果都记进 history 给下一步看。
适合“多步骤推理 / 分步任务”，比如写一份调研报告、做一道多步数学题。

两个内部角色：
    Planner  : 把问题拆成步骤列表（要求模型输出 python 列表格式）
    Executor : 按步骤逐个执行，维护执行历史
"""

import ast
from typing import Optional, List

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message


DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。请将用户的复杂问题分解成一个由多个简单步骤组成的行动计划。
每个步骤应是独立、可执行的子任务，并按逻辑顺序严格排列。
你的输出必须是一个Python列表，每个元素是一个描述子任务的字符串。

问题: {question}

请严格按以下格式输出:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。请严格按照给定计划，一步步解决问题。
你会收到原始问题、完整计划、已完成步骤的结果。请只专注解决"当前步骤"，仅输出该步答案。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class Planner:
    """规划器：负责把复杂问题拆成步骤列表。"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        response_text = self.llm_client.invoke(
            [{"role": "user", "content": prompt}], **kwargs
        ) or ""

        # 从模型输出里抠出 ```python [...] ``` 代码块，用 ast.literal_eval 安全解析成列表
        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError):
            print(f"❌ 解析计划失败，原始响应:\n{response_text}")
            return []


class Executor:
    """执行器：按计划逐步执行，维护历史。"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        history = ""
        final_answer = ""
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            response_text = self.llm_client.invoke(
                [{"role": "user", "content": prompt}], **kwargs
            ) or ""
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
        return final_answer


class PlanAndSolveAgent(Agent):
    """先规划、再分步执行的智能体。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[dict] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None
        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        final_answer = self.executor.execute(input_text, plan, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
