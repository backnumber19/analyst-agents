from typing import Dict, List, TypedDict


class AgentState(TypedDict):
    """Multi-agent workflow state"""

    # Input
    query: str
    context: str

    # Agent outputs
    research_findings: List[str]  # Research Agent
    financial_analysis: Dict[str, any]  # Financial Agent
    competitor_insights: List[str]  # Competitor Agent

    # Synthesis output
    final_report: str
    executive_summary: str
    recommendations: List[str]

    # Metadata
    iteration: int
    agent_statuses: Dict[str, str]
    errors: List[str]
