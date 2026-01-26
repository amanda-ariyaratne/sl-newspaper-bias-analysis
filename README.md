# Sri Lanka Newspaper Bias Analysis

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A data-driven analysis framework for detecting media bias in Sri Lankan English newspapers by examining coverage patterns, topic distribution, and event clustering.

## Overview

This project analyzes **8,365 articles** from **4 Sri Lankan newspapers** (Daily News, The Morning, Daily FT, The Island) covering November-December 2025 to identify:

- 📰 **Selection bias**: Which topics each source covers (or ignores)
- 🔍 **Coverage patterns**: How different sources cover the same events
- 🏷️ **Topic discovery**: Data-driven topic categorization using BERTopic
- 📊 **Event clustering**: Grouping articles about the same events across sources

## Key Findings

### Topics Discovered
- **232 topics** automatically discovered from 8,365 articles
- **77% coverage**: Successfully categorized 6,455 articles
- **Top topics**: Sri Lanka politics, flooding/disasters, sports, education, economy

### Event Clusters
- **1,717 event clusters** identified
- **87% multi-source coverage**: Most events covered by 2+ sources
- **Top event**: UN allocates $4.5M for Sri Lanka disaster relief (72 articles across 4 sources)

### Sentiment Analysis
- **Three analysis models**: Local transformer, LLM (Claude), and Hybrid approach
- **Sentiment scale**: -5 (very negative) to +5 (very positive)
- **LLM reasoning**: Detailed explanations for sentiment scores
- **Topic-sentiment correlation**: Discover how different sources frame the same topics

### Major Events (Nov-Dec 2025)
1. Cyclone Ditwah aftermath - 56 articles
2. Economic crisis response - 56 articles
3. Disaster relief fundraising - 47 articles
4. Weather warnings and flooding - multiple clusters

## Features

- 🧠 **Semantic embeddings**: 768-dimensional vectors using `all-mpnet-base-v2`
- 🎯 **Topic modeling**: BERTopic with UMAP + HDBSCAN clustering
- 🔗 **Event clustering**: Cosine similarity with time-window constraints
- 😊 **Sentiment analysis**: Three-model approach (Local, LLM, Hybrid) with reasoning
- 📈 **Interactive dashboard**: Streamlit-based visualization with 5 analysis tabs
- 🗄️ **Vector database**: PostgreSQL with pgvector extension

## Tech Stack

- **Python 3.11+**: Core language
- **PostgreSQL 16 + pgvector**: Database with vector similarity search
- **Sentence Transformers**: Local embedding generation (no API needed)
- **BERTopic**: Topic modeling with UMAP/HDBSCAN
- **Transformers + PyTorch**: Sentiment analysis with RoBERTa model
- **Claude/OpenAI API**: Optional LLM-based sentiment with reasoning
- **Streamlit**: Interactive dashboard
- **pandas, numpy**: Data processing

## Quick Start

### Prerequisites

```bash
# Database
PostgreSQL 16 with pgvector extension

# Python
Python 3.11+
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure database**
   ```bash
   # Copy configuration template
   cp config.yaml.example config.yaml

   # Edit config.yaml with your database credentials
   nano config.yaml
   ```

4. **Set up database schema**
   ```bash
   psql -h localhost -U your_db_user -d your_database -f schema.sql
   ```

5. **Run the analysis pipeline**
   ```bash
   # Generate embeddings (takes ~30 min on CPU)
   python3 scripts/01_generate_embeddings.py

   # Discover topics (takes ~2-3 min)
   python3 scripts/02_discover_topics.py

   # Cluster events (takes ~10 min)
   python3 scripts/03_cluster_events.py

   # Analyze sentiment (optional, choose model)
   python3 scripts/04_analyze_sentiment.py --model local  # Free, ~15 min
   # python3 scripts/04_analyze_sentiment.py --model llm    # $31, ~4-6 hours
   # python3 scripts/04_analyze_sentiment.py --model hybrid # $9, ~1-2 hours
   ```

6. **Launch dashboard**
   ```bash
   streamlit run dashboard/app.py
   # Access at http://localhost:8501
   ```

## Project Structure

```
sl-newspaper-bias-analysis/
├── config.yaml.example     # Configuration template
├── schema.sql              # Database schema
├── requirements.txt        # Python dependencies
├── src/
│   ├── db.py              # Database operations
│   ├── llm.py             # LLM client abstraction
│   ├── embeddings.py      # Embedding generation
│   ├── topics.py          # Topic modeling
│   ├── sentiment.py       # Sentiment analysis (3 models)
│   └── clustering.py      # Event clustering
├── scripts/
│   ├── 01_generate_embeddings.py
│   ├── 02_discover_topics.py
│   ├── 03_cluster_events.py
│   └── 04_analyze_sentiment.py
└── dashboard/
    └── app.py             # Streamlit dashboard
```

## Dashboard Preview

The dashboard includes 5 interactive views:

1. **📊 Coverage Tab**: Article volume and timeline by source
2. **🏷️ Topics Tab**: Top topics and source-topic heatmap
3. **📰 Events Tab**: Browse event clusters and cross-source coverage
4. **⚖️ Source Comparison**: Topic focus and selection bias analysis
5. **😊 Sentiment Tab**: Sentiment analysis with multiple models and visualizations

## Database Schema

### Original Data
- `news_articles` - Scraped newspaper articles (8,365 articles)

### Analysis Tables
- `embeddings` - Article embeddings (768-dim vectors)
- `topics` - Discovered topics (232 topics)
- `article_analysis` - Article-topic assignments
- `event_clusters` - Event clusters (1,717 clusters)
- `article_clusters` - Article-to-cluster mappings
- `sentiment_analyses` - Sentiment scores for each model (local/llm/hybrid)
- `sentiment_summary` - Materialized view for performance

## Sentiment Analysis

The sentiment analysis system uses three different approaches to analyze article sentiment on a scale from -5 (very negative) to +5 (very positive).

### Three Analysis Models

#### 1. Local Transformer Model (Free)
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Runtime**: ~15 minutes for 8,365 articles (CPU)
- **Cost**: Free
- **Pros**: Fast, no API needed, works offline
- **Cons**: No reasoning, less nuanced than LLM

```bash
python scripts/04_analyze_sentiment.py --model local
```

#### 2. LLM Model (Claude/OpenAI)
- **Model**: Claude Sonnet 4 (default)
- **Runtime**: ~4-6 hours for 8,365 articles
- **Cost**: ~$31 for full dataset
- **Pros**: Most accurate, provides reasoning, understands context
- **Cons**: Requires API key, costs money

```bash
# Requires ANTHROPIC_API_KEY environment variable
export ANTHROPIC_API_KEY="your-api-key"
python scripts/04_analyze_sentiment.py --model llm
```

#### 3. Hybrid Model (Best of Both)
- **Approach**: Local model + LLM fallback for low confidence
- **Runtime**: ~1-2 hours for 8,365 articles
- **Cost**: ~$9 (only ~30% use LLM)
- **Pros**: Balance of speed, cost, and accuracy
- **Cons**: Requires API key

```bash
python scripts/04_analyze_sentiment.py --model hybrid
```

### Sentiment Scale

- **-5 to -3**: Very negative (disaster, tragedy, severe criticism)
- **-2 to -1**: Somewhat negative (problems, concerns, mild criticism)
- **-0.5 to 0.5**: Neutral (factual reporting, balanced)
- **1 to 2**: Somewhat positive (progress, improvements, praise)
- **3 to 5**: Very positive (great success, celebration, strong endorsement)

### Dashboard Visualizations

The Sentiment tab in the dashboard provides:

1. **Average Sentiment by Source**: Bar chart showing which sources are more positive/negative
2. **Sentiment Distribution**: Box plots showing the range and variance
3. **Sentiment Timeline**: How sentiment changes over time
4. **Topic-Sentiment Heatmap**: Which topics are covered more positively/negatively by each source
5. **Model Comparison**: Compare agreement between different models
6. **LLM Reasoning Examples**: See detailed explanations (for LLM model)

### Cost Estimation

The script automatically estimates LLM costs before running:

```bash
python scripts/04_analyze_sentiment.py --model llm
# Shows: Estimated cost: $31.37 for 8,365 articles
# Asks for confirmation before proceeding
```

To skip the confirmation prompt:

```bash
python scripts/04_analyze_sentiment.py --model llm --skip-cost-check
```

## Research Methodology

Based on: **"The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale"** (University of Pennsylvania, 2025)

### Adaptations for Sri Lankan Context
- ✅ Topic hierarchy via data-driven discovery
- ✅ Event clustering for coverage comparison
- ✅ Selection bias analysis (topic coverage patterns)
- ✅ Sentiment analysis with multiple models
- ❌ Political lean (Democrat/Republican) - not applicable
- ⏸️ Framing bias analysis - future work

## Future Enhancements

### Planned Improvements
- **Article type classification**: news/opinion/analysis/editorial
- **Quote extraction**: Extract speaker information and attributions
- **Better topic labels**: LLM-generated descriptive topic names
- **Aspect-based sentiment**: Economic, political, social aspects
- **Entity sentiment**: Sentiment toward specific people/organizations
- **Fine-tuned local model**: Train on Sri Lankan news corpus
- **Real-time analysis**: Sentiment for newly scraped articles
- **CSV export**: Download analysis results

### Advanced Features
- Hierarchical topic relationships
- Time-series sentiment trends
- Quantified selection bias metrics
- Framing comparison across sources
- Sentiment alerts for unusual patterns

## Configuration

All configuration is in `config.yaml`:

```yaml
database:
  host: localhost
  name: your_database
  schema: your_schema
  user: your_db_user

embeddings:
  provider: local  # Free, no API needed
  model: all-mpnet-base-v2

clustering:
  similarity_threshold: 0.8
  time_window_days: 7

sentiment:
  enabled_models:
    - llm
    - local
    - hybrid

  llm_sentiment:
    provider: claude
    model: claude-sonnet-4-20250514
    batch_size: 10

  local_sentiment:
    model: cardiffnlp/twitter-roberta-base-sentiment-latest
    batch_size: 32
    device: cpu  # or cuda

  hybrid_sentiment:
    local_threshold: 0.7  # Use LLM if confidence < 0.7
```

## Performance Notes

- **Embedding generation**: ~30 minutes for 8,365 articles (CPU)
- **Topic discovery**: ~2-3 minutes
- **Event clustering**: ~10 minutes
- **Sentiment analysis**:
  - Local model: ~15 minutes (CPU)
  - LLM model: ~4-6 hours ($31)
  - Hybrid model: ~1-2 hours ($9)
- **Memory usage**: ~2GB RAM during embedding generation
- **Dashboard**: All queries cached, <3 second load times

## Troubleshooting

See [CLAUDE.md](CLAUDE.md) for detailed troubleshooting guides.

## License

MIT License - see LICENSE file for details

## Attribution

Based on the Media Bias Detector framework from University of Pennsylvania.
Adapted for Sri Lankan newspaper analysis (2025).

## Documentation

- **README.md** (this file) - Quick start and overview
- **CLAUDE.md** - Comprehensive documentation and setup guide
- **schema.sql** - Database schema with comments

## Contact

For questions or issues, please open a GitHub issue or refer to the documentation.
