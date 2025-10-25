import json
from typing import Dict

import boto3
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock


class SynthesisAgent:
    """Synthesizes insights from all agents into final report"""

    def __init__(self, config: Dict):
        self.llm = ChatBedrock(
            client=boto3.client("bedrock-runtime", region_name=config["region"]),
            model_id=config["model_id"],
            model_kwargs={"temperature": 0.2, "max_tokens": 3000},
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an executive analyst creating comprehensive battery industry reports.

Your task: Synthesize insights from specialized analysts into a cohesive report.

Report structure:
1. Executive Summary (3-4 sentences)
2. Technical Analysis (from Research Agent)
3. Financial Performance (from Financial Agent)
4. Competitive Landscape (from Competitor Agent)
5. Strategic Recommendations (3-5 bullet points)
6. Key Risks and Opportunities

Be concise, data-driven, and actionable.
""",
                ),
                (
                    "user",
                    """Original Query: {query}

Research Findings:
{research_findings}

Financial Analysis:
{financial_analysis}

Competitor Insights:
{competitor_insights}

Please synthesize these insights into a comprehensive report.""",
                ),
            ]
        )

    def synthesize(self, state: Dict) -> Dict:
        """Create final report from agent outputs"""
        try:
            # Format inputs
            research = "\n".join(state.get("research_findings", []))
            financial = json.dumps(state.get("financial_analysis", {}), indent=2)
            competitor = "\n".join(state.get("competitor_insights", []))

            # Generate report
            chain = self.prompt | self.llm
            result = chain.invoke(
                {
                    "query": state["query"],
                    "research_findings": research,
                    "financial_analysis": financial,
                    "competitor_insights": competitor,
                }
            )

            report = result.content

            # Extract executive summary (first paragraph)
            exec_summary = report.split("\n\n")[0] if report else "No summary available"

            # Extract recommendations (simplified)
            recommendations = [
                line.strip()
                for line in report.split("\n")
                if line.strip().startswith(("-", "•", "*"))
            ][:5]

            return {
                "final_report": report,
                "executive_summary": exec_summary,
                "recommendations": recommendations,
            }

        except Exception as e:
            return {
                "final_report": f"Synthesis Error: {str(e)}",
                "executive_summary": "Error generating summary",
                "recommendations": [],
            }
