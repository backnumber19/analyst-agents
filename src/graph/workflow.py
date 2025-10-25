import concurrent.futures
from typing import Dict

from langgraph.graph import END, START, StateGraph

from src.agents.competitor_agent import CompetitorIntelAgent
from src.agents.financial_agent import FinancialAnalystAgent
from src.agents.research_agent import ResearchAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.graph.state import AgentState


class MultiAgentWorkflow:
    def __init__(self, tools: Dict, config: Dict):
        self.tools = tools
        self.config = config

        self.research_agent = ResearchAgent(tools["arxiv_search"], config)
        self.financial_agent = FinancialAnalystAgent(tools["yahoo_finance"], config)
        self.competitor_agent = CompetitorIntelAgent(
            tools["news_api"], tools["arxiv_search"], config
        )
        self.synthesis_agent = SynthesisAgent(config)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("parallel_agents", self._parallel_agents_node)
        workflow.add_node("synthesis", self._synthesis_node)

        workflow.add_edge(START, "parallel_agents")
        workflow.add_edge("parallel_agents", "synthesis")
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    def _parallel_agents_node(self, state: AgentState) -> Dict:
        def run_research():
            try:
                return self.research_agent.analyze(
                    state["query"], state.get("context", "")
                )
            except Exception as e:
                return [f"Research error: {str(e)}"]

        def run_financial():
            try:
                return self.financial_agent.analyze(
                    state["query"], state.get("context", "")
                )
            except Exception as e:
                return {"error": f"Financial error: {str(e)}"}

        def run_competitor():
            try:
                return self.competitor_agent.analyze(
                    state["query"], state.get("context", "")
                )
            except Exception as e:
                return [f"Competitor error: {str(e)}"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_research = executor.submit(run_research)
            future_financial = executor.submit(run_financial)
            future_competitor = executor.submit(run_competitor)

            research_findings = future_research.result()
            financial_analysis = future_financial.result()
            competitor_insights = future_competitor.result()

        return {
            "research_findings": research_findings,
            "financial_analysis": financial_analysis,
            "competitor_insights": competitor_insights,
        }

    def _synthesis_node(self, state: AgentState) -> Dict:
        report_data = self.synthesis_agent.synthesize(state)
        return {
            **report_data,
            "agent_statuses": {
                "research": "completed",
                "financial": "completed",
                "competitor": "completed",
                "synthesis": "completed",
            },
        }

    def run(self, query: str, context: str = "") -> Dict:
        initial_state = {
            "query": query,
            "context": context,
            "research_findings": [],
            "financial_analysis": {},
            "competitor_insights": [],
            "final_report": "",
            "executive_summary": "",
            "recommendations": [],
            "iteration": 0,
            "agent_statuses": {},
            "errors": [],
        }

        result = self.graph.invoke(initial_state)
        return result
