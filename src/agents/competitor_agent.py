from typing import Dict, List

import boto3
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock


class CompetitorIntelAgent:

    def __init__(self, news_tool, research_tool, config: Dict):
        self.llm = ChatBedrock(
            client=boto3.client("bedrock-runtime", region_name=config["region"]),
            model_id=config["model_id"],
            model_kwargs={"temperature": 0.1, "max_tokens": 2000},
        )

        self.tools = [news_tool, research_tool]

        self.prompt = PromptTemplate.from_template(
            """
You are a competitive intelligence analyst for the battery industry.

Use the following tools to monitor competitor activities and market trends.

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
Thought: {agent_scratchpad}
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
        full_query = f"{query}\n\nContext: {context}" if context else query

        try:
            result = self.executor.invoke({"input": full_query})
            output = result.get("output", "")

            insights = []
            for line in output.split("\n"):
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("•") or line.startswith("*")
                ):
                    insights.append(line.lstrip("-•* "))

            return insights if insights else [output]

        except Exception as e:
            return [f"Competitor Agent Error: {str(e)}"]
