import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.financial_agent import FinancialAnalystAgent
from tools.finance_api import FinanceDataTool


class TestFinancialAgent:
    """Test Financial Agent functionality"""

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
        finance_tool = FinanceDataTool().as_langchain_tool()
        agent = FinancialAnalystAgent(finance_tool, config)

        assert agent is not None
        assert hasattr(agent, "llm")
        assert hasattr(agent, "tools")
        assert hasattr(agent, "executor")
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "yahoo_finance"

    def test_analyze_method(self, config):
        """Test analyze method returns correct format"""
        finance_tool = FinanceDataTool().as_langchain_tool()
        agent = FinancialAnalystAgent(finance_tool, config)

        result = agent.analyze("test query")

        assert isinstance(result, dict)
        assert "analysis" in result
        assert "status" in result
        assert result["status"] in ["success", "error"]

    def test_analyze_with_context(self, config):
        """Test analyze method with context"""
        finance_tool = FinanceDataTool().as_langchain_tool()
        agent = FinancialAnalystAgent(finance_tool, config)

        query = "LG Energy Solution financials"
        context = "focus on 2024 performance"

        result = agent.analyze(query, context)

        assert isinstance(result, dict)
        assert "analysis" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
