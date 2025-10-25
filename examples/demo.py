import os
import sys
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_VERBOSE"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

print("=" * 80)
print("BATTERY ANALYST AGENTS - INTERACTIVE DEMO")
print("=" * 80)

print("\nChecking connections...")
try:
    import boto3

    sts = boto3.client("sts", region_name="us-west-2")
    identity = sts.get_caller_identity()
    print(f"AWS: Connected as {identity['Arn'].split('/')[-1]}")
except Exception as e:
    print(f"AWS: Connection failed - {e}")
    sys.exit(1)

api_key = os.getenv("NEWS_API_KEY")
if api_key and api_key != "your-news-api-key-here":
    print("News API: Configured")
else:
    print("News API: Not configured")
    sys.exit(1)

print("\nInitializing agents...")
try:
    from src.agents.competitor_agent import CompetitorIntelAgent
    from src.agents.financial_agent import FinancialAnalystAgent
    from src.agents.research_agent import ResearchAgent
    from src.agents.synthesis_agent import SynthesisAgent
    from src.tools.arxiv_search import ArxivSearchTool
    from src.tools.finance_api import FinanceDataTool
    from src.tools.news_api import NewsSearchTool

    config = {
        "region": os.getenv("AWS_REGION", "us-west-2"),
        "model_id": os.getenv(
            "BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ),
    }

    arxiv_tool = ArxivSearchTool().as_langchain_tool()
    finance_tool = FinanceDataTool().as_langchain_tool()
    news_tool = NewsSearchTool(api_key).as_langchain_tool()
    tools = {
        "arxiv_search": arxiv_tool,
        "yahoo_finance": finance_tool,
        "news_api": news_tool,
    }

    research_agent = ResearchAgent(arxiv_tool, config)
    financial_agent = FinancialAnalystAgent(finance_tool, config)
    competitor_agent = CompetitorIntelAgent(news_tool, arxiv_tool, config)
    synthesis_agent = SynthesisAgent(config)

    print("All agents initialized successfully")

except Exception as e:
    print(f"Initialization failed: {e}")
    sys.exit(1)

# Example prompts
EXAMPLES = """
Example queries:

### Technology Analysis
1. "What are the latest developments in AI/blockchain/quantum computing?"
2. "How do different cloud providers compare in terms of performance and cost?"
3. "What are the key challenges in scaling up renewable energy?"

### Financial Analysis
4. "Analyze Tesla's financial performance and market position"
5. "Compare Microsoft and Google's revenue growth and profitability"
6. "What is the market outlook for semiconductor companies in 2024?"

### Competitive Intelligence
7. "How do US tech companies compare to Chinese competitors?"
8. "What are the recent competitive developments in the electric vehicle market?"
9. "Who are the emerging players in the fintech industry?"

### Strategic Analysis
10. "What are the key risks and opportunities for Apple in the smartphone industry?"
11. "How should European companies respond to US tech dominance?"
12. "What partnerships should Amazon pursue for global expansion?"

CUSTOM QUERIES:
- You can ask any question about battery technology, companies, or market trends
- Be specific about companies, technologies, or timeframes for better results
"""

print("\n" + "=" * 80)
print(EXAMPLES)
print("=" * 80)


def run_analysis(query):
    """Run multi-agent analysis with LangGraph async processing"""
    print(f"\nQuery: {query}")
    print("\n" + "-" * 80)
    print("Running async analysis...")
    print("-" * 80)

    import logging

    logging.getLogger().setLevel(logging.ERROR)

    try:
        import time

        start_time = time.time()

        from src.graph.workflow import MultiAgentWorkflow

        workflow = MultiAgentWorkflow(tools, config)

        print("\nStarting parallel analysis...")

        result = workflow.run(query)

        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.1f}s")

        print("\n" + "=" * 80)
        print("ANALYSIS REPORT")
        print("=" * 80)

        print("\nEXECUTIVE SUMMARY:")
        print("-" * 80)
        print(result["executive_summary"])

        print("\n\nFULL REPORT:")
        print("-" * 80)
        print(result["final_report"])

        print("\n\nKEY RECOMMENDATIONS:")
        print("-" * 80)
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 80)

        save_option = input("\nSave report to file? (y/n): ").strip().lower()
        if save_option == "y":
            filename = f"report_{query[:30].replace(' ', '_').replace('?', '')}.txt"
            with open(filename, "w") as f:
                f.write(f"Query: {query}\n\n")
                f.write("=" * 80 + "\n")
                f.write("ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write("EXECUTIVE SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(result["executive_summary"] + "\n\n")
                f.write("FULL REPORT:\n")
                f.write("-" * 80 + "\n")
                f.write(result["final_report"] + "\n\n")
                f.write("KEY RECOMMENDATIONS:\n")
                f.write("-" * 80 + "\n")
                for i, rec in enumerate(result["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
            print(f"\nReport saved to: {filename}")

    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()


while True:  # interactive loop
    print("\n" + "=" * 80)
    query = input(
        "\nEnter your query (or 'examples' to see examples, 'quit' to exit): "
    ).strip()

    if not query:
        continue

    if query.lower() == "quit":
        print("\nExiting...")
        break

    if query.lower() == "examples":
        print(EXAMPLES)
        continue

    run_analysis(query)

    print("\n" + "=" * 80)
    continue_option = input("\nAnalyze another query? (y/n): ").strip().lower()
    if continue_option != "y":
        print("\nExiting...")
        break

print("\nThank you for using Analyst Agents!")
