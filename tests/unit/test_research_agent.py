import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.research_agent import ResearchAgent
from tools.arxiv_search import ArxivSearchTool


class TestResearchAgent:
    """Test Research Agent functionality"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        load_dotenv()
        return {
            "region": os.getenv("AWS_REGION", "us-west-2"),
            "model_id": os.getenv(
                "BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
        }

    def test_agent_initialization(self, config):
        """Test agent initializes correctly"""
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = ResearchAgent(arxiv_tool, config)

        assert agent is not None
        assert hasattr(agent, "llm")
        assert hasattr(agent, "tools")
        assert hasattr(agent, "executor")
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "arxiv_search"

    def test_analyze_method_signature(self, config):
        """Test analyze method accepts correct parameters"""
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = ResearchAgent(arxiv_tool, config)

        result = agent.analyze("test query")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_analyze_with_context(self, config):
        """Test analyze method with context"""
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = ResearchAgent(arxiv_tool, config)

        query = "battery technology"
        context = "focus on lithium-ion"

        result = agent.analyze(query, context)

        assert isinstance(result, list)

    def test_error_handling(self, config):
        """Test error handling in analyze method"""
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = ResearchAgent(arxiv_tool, config)

        result = agent.analyze("")
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
