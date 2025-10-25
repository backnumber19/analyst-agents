import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.competitor_agent import CompetitorIntelAgent
from tools.arxiv_search import ArxivSearchTool
from tools.news_api import NewsSearchTool


class TestCompetitorAgent:
    """Test Competitor Agent functionality"""

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
        news_tool = NewsSearchTool("test-key").as_langchain_tool()
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = CompetitorIntelAgent(news_tool, arxiv_tool, config)

        assert agent is not None
        assert hasattr(agent, "llm")
        assert hasattr(agent, "tools")
        assert hasattr(agent, "executor")
        assert len(agent.tools) == 2

    def test_analyze_method(self, config):
        """Test analyze method returns list"""
        news_tool = NewsSearchTool("test-key").as_langchain_tool()
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = CompetitorIntelAgent(news_tool, arxiv_tool, config)

        result = agent.analyze("test query")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_analyze_with_context(self, config):
        """Test analyze method with context"""
        news_tool = NewsSearchTool("test-key").as_langchain_tool()
        arxiv_tool = ArxivSearchTool().as_langchain_tool()
        agent = CompetitorIntelAgent(news_tool, arxiv_tool, config)

        query = "battery industry competition"
        context = "focus on Korean companies"

        result = agent.analyze(query, context)

        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
