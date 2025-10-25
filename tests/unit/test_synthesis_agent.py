import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.synthesis_agent import SynthesisAgent


class TestSynthesisAgent:
    """Test Synthesis Agent functionality"""

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

    @pytest.fixture
    def mock_state(self):
        """Mock state for testing"""
        return {
            "query": "What is LG Energy Solution's competitive position?",
            "research_findings": [
                "Advanced NCM battery technology",
                "Strong R&D investment in solid-state batteries",
                "Partnership with major automakers",
            ],
            "financial_analysis": {
                "revenue_trends": "Growing revenue in EV battery segment",
                "profitability": "Improving margins due to scale",
                "market_position": "Strong position in premium EV market",
            },
            "competitor_insights": [
                "CATL leads in market share but LG has technology edge",
                "Samsung SDI focusing on cylindrical batteries",
                "BYD expanding globally with competitive pricing",
            ],
        }

    def test_agent_initialization(self, config):
        """Test agent initializes correctly"""
        agent = SynthesisAgent(config)

        assert agent is not None
        assert hasattr(agent, "llm")
        assert hasattr(agent, "prompt")

    def test_synthesize_method(self, config, mock_state):
        """Test synthesize method returns correct format"""
        agent = SynthesisAgent(config)
        result = agent.synthesize(mock_state)

        assert isinstance(result, dict)
        assert "final_report" in result
        assert "executive_summary" in result
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_synthesize_with_empty_state(self, config):
        """Test synthesize with empty state"""
        agent = SynthesisAgent(config)

        empty_state = {
            "query": "",
            "research_findings": [],
            "financial_analysis": {},
            "competitor_insights": [],
        }

        result = agent.synthesize(empty_state)

        assert isinstance(result, dict)
        assert "final_report" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
