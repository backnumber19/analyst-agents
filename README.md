# Multi-Agent Industry Analyst System 🤖

Multi-agent system for comprehensive industry analysis using LangGraph, AWS Bedrock, and external APIs.

## TL;DR

A specialized multi-agent RAG system that analyzes battery industry through parallel AI agents. Each agent handles different aspects: technical research (arXiv), financial analysis (Yahoo Finance), and competitive intelligence (News API). Uses LangGraph for orchestration and AWS Bedrock (Claude 3 Haiku) for reasoning.

**Tech Stack**: LangGraph, AWS Bedrock, Claude 3 Haiku, arXiv API, Yahoo Finance, News API, Python

## What Makes This Different from Traditional RAG

| Feature | Traditional RAG | Multi-Agent RAG |
|---------|----------------|-----------------|
| **Data Source** | Static documents | Real-time APIs |
| **Processing** | Single retrieval | Parallel specialized agents |
| **Intelligence** | Document search | Active research & analysis |
| **Scope** | Pre-indexed content | Dynamic market intelligence |
| **Output** | Document-based answers | Comprehensive industry reports |

## Project Structure

```text
analyst-agents/
├── examples/
│   └── demo.py                    # Interactive demo
├── src/
│   ├── agents/                    # Specialized AI agents
│   │   ├── research_agent.py      # Technical research
│   │   ├── financial_agent.py     # Financial analysis
│   │   ├── competitor_agent.py    # Competitive intelligence
│   │   └── synthesis_agent.py     # Report synthesis
│   ├── tools/                     # API integrations
│   │   ├── arxiv_search.py        # Academic papers
│   │   ├── finance_api.py         # Yahoo Finance
│   │   └── news_api.py            # News API
│   └── graph/
│       ├── state.py               # Shared state
│       └── workflow.py            # LangGraph orchestration
├── tests/
│   ├── unit/                      # Agent unit tests
│   └── test_setup.py              # Connection tests
├── report_result_example.txt      # Example report
├── .env.example                   # Example .env file
├── requirements.txt
└── README.md
```

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                 Multi-Agent Reasoning System                 │
└──────────────────────────────────────────────────────────────┘

User Query: "Analyze solid-state battery developments and LG Energy Solution's position"
    ↓
┌──────────────────────────────────────────────────────────────┐
│                        Orchestrator                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Script: src/graph/workflow.py                          │ │
│  │  Function: run() -> _parallel_agents_node()             │ │
│  │                                                         │ │
│  │  - Simple query pass-through                            │ │
│  │  - Direct to parallel agent execution                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│                Parallel Agent Execution                      │
│                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │   Research  │ │  Financial  │ │ Competitor  │             │
│  │    Agent    │ │    Agent    │ │    Agent    │             │
│  │             │ │             │ │             │             │
│  │ Script:     │ │ Script:     │ │ Script:     │             │
│  │ src/agents/ │ │ src/agents/ │ │ src/agents/ │             │
│  │ research_   │ │ financial_  │ │ competitor_ │             │
│  │ agent.py    │ │ agent.py    │ │ agent.py    │             │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘             │
│         │               │               │                    │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐             │
│  │   arXiv     │ │   Yahoo     │ │   News      │             │
│  │    API      │ │  Finance    │ │    API      │             │
│  │             │ │             │ │             │             │
│  │ Script:     │ │ Script:     │ │ Script:     │             │
│  │ src/tools/  │ │ src/tools/  │ │ src/tools/  │             │
│  │ arxiv_      │ │ finance_    │ │ news_api.py │             │
│  │ search.py   │ │ api.py      │ │             │             │
│  └─────────────┘ └─────────────┘ └─────────────┘             │
└──────────────────────────────────────────────────────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
┌──────────────────────────────────────────────────────────────┐
│                      State Aggregation                       │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Script: src/graph/workflow.py                          │ │
│  │  Function: _parallel_agents_node()                      │ │
│  │                                                         │ │
│  │  research_findings: [                                   │ │
│  │    "Solid-state batteries show 40% energy density...",  │ │
│  │    "New cathode materials enable faster charging..."    │ │
│  │  ]                                                      │ │
│  │                                                         │ │
│  │  financial_analysis: {                                  │ │
│  │    "lg_energy": {"revenue": "$2.1B", "growth": "15%"},  │ │
│  │    "competitors": {...}                                 │ │
│  │  }                                                      │ │
│  │                                                         │ │
│  │  competitor_insights: [                                 │ │
│  │    "CATL announces new factory in Europe...",           │ │
│  │    "BYD partners with Tesla for battery supply..."      │ │
│  │  ]                                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│                       Synthesis Agent                        │ 
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Script: src/agents/synthesis_agent.py                  │ │
│  │  Function: synthesize(state)                            │ │
│  │                                                         │ │
│  │  Reasoning Process:                                     │ │
│  │                                                         │ │
│  │  1. Analyze research findings                           │ │
│  │     → "Latest solid-state tech shows 40% improvement"   │ │
│  │                                                         │ │
│  │  2. Evaluate financial data                             │ │
│  │     → "LG Energy Solution leads in revenue growth"      │ │
│  │                                                         │ │
│  │  3. Assess competitive landscape                        │ │
│  │     → "Chinese competitors gaining market share"        │ │
│  │                                                         │ │
│  │  4. Identify patterns and insights                      │ │
│  │     → "Technology gap between Korea and China"          │ │
│  │                                                         │ │
│  │  5. Generate strategic recommendations                  │ │
│  │     → "Focus on R&D investment in solid-state tech"     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: Executive Report with Actionable Insights           │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- News API key (free tier available)

### 1. Clone and Setup

```bash
cd /path/to/analyst-agents

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-west-2
AWS_PROFILE=default

# Bedrock Model
BEDROCK_MODEL=anthropic.claude-3-haiku-20240307-v1:0

# News API
NEWS_API_KEY=your-news-api-key-here

# Model Settings
TEMPERATURE=0.1
MAX_TOKENS=2000
```

### 3. Get API Keys

#### News API (Free)
1. Go to: https://newsapi.org/
2. Sign up for free account
3. Get API key

#### AWS Bedrock
1. Go to: https://console.aws.amazon.com/bedrock/
2. Request access to Claude 3 Haiku
3. Configure AWS CLI: `aws configure`

### 4. Run Demo

```bash
# Interactive demo
python examples/demo.py

# Example queries:
# "What are the latest developments in solid-state battery technology? 
# And what are the solid-state battery product strategy of LG Energy Solution and its chinese competitors? 
# Lastly, Analyze LG Energy Solution's financial performance and market position."
```
→ Generated report: report_result_example.txt

## Agent Capabilities

### Research Agent 🔬
- **Tool**: arXiv API
- **Capability**: Academic paper analysis
- **Output**: Technical insights, research trends
- **Example**: "Latest AI research from top universities"

### Financial Agent 💰
- **Tool**: Yahoo Finance API
- **Capability**: Company financial analysis
- **Output**: Revenue, profitability, market position
- **Example**: "Tesla vs Toyota financial comparison"

### Competitor Agent 🏢
- **Tool**: News API + arXiv
- **Capability**: Industry intelligence
- **Output**: Market developments, competitive moves
- **Example**: "Recent AI industry partnerships and acquisitions"

### Synthesis Agent
- **Capability**: Report generation
- **Output**: Executive summaries, recommendations
- **Example**: "Comprehensive industry analysis with strategic insights"

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Full report generation | 15-20 sec | Including synthesis |

*Based on example query in 4. Run Demo


## Cost Estimate

| Service | Usage | Cost/Query | Cost/Month* |
|---------|-------|------------|-------------|
| Claude 3 Haiku | 2K tokens | ~$0.0005 | ~$1.50 |
| News API | 100 requests | Free | $0 |
| arXiv API | Unlimited | Free | $0 |
| Yahoo Finance API | Unlimited | Free | $0 |
| **Total** | | **~$0.0005** | **~$1.50** |

*Based on 100 queries/day

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test individual agents
pytest tests/unit/ -v

# Test connections
python tests/test_setup.py
```

## Why This is "Agent-Based"

1. **Autonomous Decision Making**: Each agent decides which tools to use and how
2. **Specialized Reasoning**: Domain-specific analysis capabilities
3. **Tool Integration**: Seamless API usage for real-time data
4. **Collaborative Intelligence**: Agents work together to create comprehensive insights
5. **Adaptive Processing**: Different agents for different types of analysis

This system goes beyond simple RAG by providing **active intelligence gathering** and **specialized analysis** rather than just document retrieval.