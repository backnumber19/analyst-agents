from typing import Dict, List

import boto3
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock


class ResearchAgent:
    """Technical research and patent analysis agent"""

    def __init__(self, research_tool, config: Dict):
        self.llm = ChatBedrock(
            client=boto3.client("bedrock-runtime", region_name=config["region"]),
            model_id=config["model_id"],
            model_kwargs={"temperature": 0.1, "max_tokens": 2000},
        )

        self.tools = [research_tool]

        # 최신 방식: create_react_agent 사용
        self.prompt = PromptTemplate.from_template(
            """
You are a technical research analyst specializing in battery technology.

Use the following tools to answer questions. Be concise and technical.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
        )

        self.agent = create_react_agent(
            llm=self.llm, tools=self.tools, prompt=self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True,
        )

    def analyze(self, query: str, context: str = "") -> List[str]:
        """Run research analysis"""
        full_query = f"{query}\n\nContext: {context}" if context else query

        try:
            result = self.executor.invoke({"input": full_query})
            output = result.get("output", "")

            # Parse findings
            findings = []
            for line in output.split("\n"):
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("•") or line.startswith("*")
                ):
                    findings.append(line.lstrip("-•* "))

            return findings if findings else [output]

        except Exception as e:
            return [f"Research Agent Error: {str(e)}"]
