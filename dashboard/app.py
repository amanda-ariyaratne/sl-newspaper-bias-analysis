"""Sri Lanka Media Bias Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_db
from bertopic import BERTopic

# Page config
st.set_page_config(
    page_title="Sri Lanka Media Bias Detector",
    page_icon="📰",
    layout="wide"
)

# Source name mapping
SOURCE_NAMES = {
    "dailynews_en": "Daily News",
    "themorning_en": "The Morning",
    "ft_en": "Daily FT",
    "island_en": "The Island"
}

SOURCE_COLORS = {
    "Daily News": "#1f77b4",
    "The Morning": "#ff7f0e",
    "Daily FT": "#2ca02c",
    "The Island": "#d62728"
}


@st.cache_data(ttl=300)
def load_overview_stats():
    """Load overview statistics."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Total articles
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.news_articles")
            total_articles = cur.fetchone()["count"]

            # Articles by source
            cur.execute(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.news_articles
                GROUP BY source_id
                ORDER BY count DESC
            """)
            by_source = cur.fetchall()

            # Total topics
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.topics WHERE topic_id != -1")
            total_topics = cur.fetchone()["count"]

            # Total clusters
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.event_clusters")
            total_clusters = cur.fetchone()["count"]

            # Multi-source clusters
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE sources_count > 1")
            multi_source = cur.fetchone()["count"]

            # Date range
            cur.execute(f"""
                SELECT MIN(date_posted)::date as min_date, MAX(date_posted)::date as max_date
                FROM {schema}.news_articles
            """)
            date_range = cur.fetchone()

    return {
        "total_articles": total_articles,
        "by_source": by_source,
        "total_topics": total_topics,
        "total_clusters": total_clusters,
        "multi_source_clusters": multi_source,
        "date_range": date_range
    }


@st.cache_data(ttl=300)
def load_topics():
    """Load topic data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT topic_id, name, description, article_count
                FROM {schema}.topics
                WHERE topic_id != -1
                ORDER BY article_count DESC
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_by_source(model_type: str):
    """Load average sentiment by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment,
                    STDDEV(sa.overall_sentiment) as stddev_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
                GROUP BY n.source_id
                ORDER BY avg_sentiment DESC
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_distribution(model_type: str):
    """Load sentiment distribution for box plots."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.source_id,
                    sa.overall_sentiment
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_percentage_by_source(model_type: str):
    """Load sentiment percentage distribution by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.source_id,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment < -0.5) as negative_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment >= -0.5 AND sa.overall_sentiment <= 0.5) as neutral_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment > 0.5) as positive_count,
                    COUNT(*) as total_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
                GROUP BY n.source_id
                ORDER BY n.source_id
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_timeline(model_type: str):
    """Load sentiment over time."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    DATE_TRUNC('day', n.date_posted) as date,
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
                GROUP BY DATE_TRUNC('day', n.date_posted), n.source_id
                ORDER BY date
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_sentiment(model_type: str):
    """Load sentiment by topic."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    t.name as topic,
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE sa.model_type = %s AND t.topic_id != -1
                GROUP BY t.name, n.source_id
                HAVING COUNT(*) >= 5
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_available_models():
    """Get list of models with analysis results."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT model_type, COUNT(*) as article_count
                FROM {schema}.sentiment_analyses
                GROUP BY model_type
                ORDER BY model_type
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_list():
    """Get list of topics for dropdown."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT name, article_count
                FROM {schema}.topics
                WHERE topic_id != -1
                ORDER BY article_count DESC
                LIMIT 50
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_by_source_topic(model_type: str, topic: str = None):
    """Load sentiment by source, optionally filtered by topic."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            query = f"""
                SELECT
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment,
                    STDDEV(sa.overall_sentiment) as stddev_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
            """

            if topic and topic != "All Topics":
                query += f"""
                    JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                    JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                    WHERE sa.model_type = %s AND t.name = %s
                """
                params = (model_type, topic)
            else:
                query += " WHERE sa.model_type = %s"
                params = (model_type,)

            query += " GROUP BY n.source_id ORDER BY avg_sentiment DESC"

            cur.execute(query, params)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_percentage_by_source_topic(model_type: str, topic: str = None):
    """Load sentiment percentage distribution by source with optional topic filter."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            query = f"""
                SELECT
                    n.source_id,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment < -0.5) as negative_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment >= -0.5 AND sa.overall_sentiment <= 0.5) as neutral_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment > 0.5) as positive_count,
                    COUNT(*) as total_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
            """

            if topic and topic != "All Topics":
                query += f"""
                    JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                    JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                    WHERE sa.model_type = %s AND t.name = %s
                """
                params = (model_type, topic)
            else:
                query += " WHERE sa.model_type = %s"
                params = (model_type,)

            query += " GROUP BY n.source_id ORDER BY n.source_id"

            cur.execute(query, params)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_multi_model_comparison(models: list, topic: str = None):
    """Load sentiment data for multiple models with optional topic filter."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            query = f"""
                SELECT
                    sa.model_type,
                    n.source_id,
                    sa.overall_sentiment,
                    t.name as topic
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                LEFT JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                LEFT JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE sa.model_type = ANY(%s)
            """
            params = [models]

            if topic and topic != "All Topics":
                query += " AND t.name = %s"
                params.append(topic)

            cur.execute(query, tuple(params))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_by_source():
    """Load topic distribution by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT t.name as topic, n.source_id, COUNT(*) as count
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.topic_id
                JOIN {schema}.news_articles n ON aa.article_id = n.id
                WHERE t.topic_id != -1
                GROUP BY t.name, n.source_id
                ORDER BY count DESC
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_top_events(limit=20):
    """Load top event clusters."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id, ec.cluster_name, ec.article_count, ec.sources_count,
                       ec.date_start, ec.date_end
                FROM {schema}.event_clusters ec
                ORDER BY ec.article_count DESC
                LIMIT {limit}
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_event_details(event_id):
    """Load details for a specific event cluster."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Get articles in cluster
            cur.execute(f"""
                SELECT n.title, n.source_id, n.date_posted, n.url
                FROM {schema}.article_clusters ac
                JOIN {schema}.news_articles n ON ac.article_id = n.id
                WHERE ac.cluster_id = %s
                ORDER BY n.date_posted
            """, (event_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_coverage_timeline():
    """Load daily article counts by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT date_posted::date as date, source_id, COUNT(*) as count
                FROM {schema}.news_articles
                WHERE date_posted IS NOT NULL
                GROUP BY date_posted::date, source_id
                ORDER BY date
            """)
            return cur.fetchall()


@st.cache_resource
def load_bertopic_model():
    """Load the saved BERTopic model."""
    model_path = Path(__file__).parent.parent / "models" / "bertopic_model"
    if model_path.exists():
        try:
            return BERTopic.load(str(model_path))
        except Exception as e:
            st.warning(f"Could not load BERTopic model: {e}")
            return None
    return None


def main():
    st.title("🇱🇰 Sri Lanka Media Bias Detector")
    st.markdown("Analyzing coverage patterns across Sri Lankan English newspapers")

    # Load data
    stats = load_overview_stats()

    # Overview metrics
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Articles", f"{stats['total_articles']:,}")
    with col2:
        st.metric("Topics Discovered", stats['total_topics'])
    with col3:
        st.metric("Event Clusters", f"{stats['total_clusters']:,}")
    with col4:
        st.metric("Multi-Source Events", f"{stats['multi_source_clusters']:,}")

    if stats['date_range']['min_date']:
        st.caption(f"Date range: {stats['date_range']['min_date']} to {stats['date_range']['max_date']}")

    st.divider()

    # Initialize session state for active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Tabs for different views
    tab_names = ["📊 Coverage", "🏷️ Topics", "📰 Events", "⚖️ Source Comparison", "😊 Sentiment"]

    # Create buttons to switch tabs
    cols = st.columns(5)
    for idx, (col, tab_name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(tab_name, key=f"tab_{idx}",
                        type="primary" if st.session_state.active_tab == idx else "secondary"):
                st.session_state.active_tab = idx

    st.divider()

    # Render the active tab
    if st.session_state.active_tab == 0:
        render_coverage_tab(stats)
    elif st.session_state.active_tab == 1:
        render_topics_tab()
    elif st.session_state.active_tab == 2:
        render_events_tab()
    elif st.session_state.active_tab == 3:
        render_comparison_tab()
    elif st.session_state.active_tab == 4:
        render_sentiment_tab()


def render_coverage_tab(stats):
    """Render coverage analysis tab."""
    st.subheader("Article Coverage by Source")

    # Articles by source bar chart
    source_df = pd.DataFrame(stats['by_source'])
    source_df['source_name'] = source_df['source_id'].map(SOURCE_NAMES)

    fig = px.bar(
        source_df,
        x='source_name',
        y='count',
        color='source_name',
        color_discrete_map=SOURCE_COLORS,
        labels={'count': 'Articles', 'source_name': 'Source'}
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.subheader("Coverage Over Time")
    timeline_data = load_coverage_timeline()

    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['source_name'] = timeline_df['source_id'].map(SOURCE_NAMES)

        fig = px.line(
            timeline_df,
            x='date',
            y='count',
            color='source_name',
            color_discrete_map=SOURCE_COLORS,
            labels={'count': 'Articles', 'date': 'Date', 'source_name': 'Source'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_topics_tab():
    """Render topics analysis tab."""
    st.subheader("Discovered Topics")

    topics = load_topics()
    topics_df = pd.DataFrame(topics)

    # Top 20 topics bar chart
    top_topics = topics_df.head(20)

    fig = px.bar(
        top_topics,
        x='article_count',
        y='name',
        orientation='h',
        labels={'article_count': 'Articles', 'name': 'Topic'}
    )
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Topic by source heatmap
    st.subheader("Topic Coverage by Source")

    topic_source_data = load_topic_by_source()
    if topic_source_data:
        ts_df = pd.DataFrame(topic_source_data)
        ts_df['source_name'] = ts_df['source_id'].map(SOURCE_NAMES)

        # Get top 15 topics for heatmap
        top_topic_names = topics_df.head(15)['name'].tolist()
        ts_filtered = ts_df[ts_df['topic'].isin(top_topic_names)]

        # Pivot for heatmap
        pivot_df = ts_filtered.pivot(index='topic', columns='source_name', values='count').fillna(0)

        fig = px.imshow(
            pivot_df,
            labels=dict(x="Source", y="Topic", color="Articles"),
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # BERTopic Visualizations
    st.divider()
    st.subheader("Topic Model Visualizations")

    topic_model = load_bertopic_model()
    if topic_model:
        viz_option = st.selectbox(
            "Select visualization",
            [
                "Topic Similarity Map (2D)",
                "Topic Bar Charts",
                "Topic Similarity Heatmap",
                "Hierarchical Topic Clustering"
            ]
        )

        try:
            if viz_option == "Topic Similarity Map (2D)":
                st.markdown("**Interactive 2D visualization of topic relationships**")
                st.caption("Topics closer together are more semantically similar")
                fig = topic_model.visualize_topics()
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Topic Bar Charts":
                st.markdown("**Top words per topic**")
                # Show top 10 topics
                top_topics_ids = [t['topic_id'] for t in topics[:10]]
                fig = topic_model.visualize_barchart(top_n_topics=10, topics=top_topics_ids)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Topic Similarity Heatmap":
                st.markdown("**Similarity matrix between topics**")
                st.caption("Darker colors indicate higher similarity")
                # Limit to top 20 topics for readability
                top_topics_ids = [t['topic_id'] for t in topics[:20]]
                fig = topic_model.visualize_heatmap(topics=top_topics_ids)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Hierarchical Topic Clustering":
                st.markdown("**Hierarchical clustering of topics**")
                st.caption("Shows how topics group into broader categories")
                fig = topic_model.visualize_hierarchy()
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.info("BERTopic model not found. Run `python3 scripts/02_discover_topics.py` to generate the model.")


def render_events_tab():
    """Render events analysis tab."""
    st.subheader("Top Event Clusters")
    st.markdown("Events covered by multiple sources - useful for comparing coverage")

    events = load_top_events(30)
    events_df = pd.DataFrame(events)

    # Filter to multi-source events
    multi_source_events = events_df[events_df['sources_count'] > 1]

    # Event selector
    event_options = {
        f"{e['cluster_name'][:60]}... ({e['article_count']} articles, {e['sources_count']} sources)": e['id']
        for _, e in multi_source_events.iterrows()
    }

    selected_event_label = st.selectbox(
        "Select an event to explore",
        options=list(event_options.keys())
    )

    if selected_event_label:
        event_id = event_options[selected_event_label]
        articles = load_event_details(event_id)

        if articles:
            articles_df = pd.DataFrame(articles)
            articles_df['source_name'] = articles_df['source_id'].map(SOURCE_NAMES)

            # Source breakdown
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Coverage by Source**")
                source_counts = articles_df['source_name'].value_counts()

                fig = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    color=source_counts.index,
                    color_discrete_map=SOURCE_COLORS
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Articles in this Event**")
                display_df = articles_df[['title', 'source_name', 'date_posted']].copy()
                display_df.columns = ['Title', 'Source', 'Date']
                st.dataframe(display_df, use_container_width=True, height=300)


def render_comparison_tab():
    """Render source comparison tab."""
    st.subheader("Source Comparison")
    st.markdown("Compare how different sources cover the same topics and events")

    # Topic coverage comparison
    st.markdown("### Topic Focus by Source")
    st.markdown("What percentage of each source's coverage goes to each topic?")

    topic_source_data = load_topic_by_source()
    if topic_source_data:
        ts_df = pd.DataFrame(topic_source_data)
        ts_df['source_name'] = ts_df['source_id'].map(SOURCE_NAMES)

        # Calculate percentages per source
        source_totals = ts_df.groupby('source_name')['count'].sum()

        # Get top 10 topics
        topics = load_topics()
        top_topic_names = [t['name'] for t in topics[:10]]

        comparison_data = []
        for source in SOURCE_NAMES.values():
            source_data = ts_df[ts_df['source_name'] == source]
            total = source_totals.get(source, 1)

            for topic in top_topic_names:
                topic_count = source_data[source_data['topic'] == topic]['count'].sum()
                comparison_data.append({
                    'Source': source,
                    'Topic': topic[:30],
                    'Percentage': (topic_count / total) * 100
                })

        comp_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comp_df,
            x='Topic',
            y='Percentage',
            color='Source',
            barmode='group',
            color_discrete_map=SOURCE_COLORS,
            labels={'Percentage': '% of Coverage'}
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Selection bias indicator
    st.markdown("### Selection Bias Indicators")
    st.markdown("Topics where sources significantly differ in coverage")

    if topic_source_data:
        # Calculate variance in coverage percentage across sources
        variance_data = []
        for topic in top_topic_names:
            topic_data = comp_df[comp_df['Topic'] == topic[:30]]
            if len(topic_data) > 1:
                variance_data.append({
                    'Topic': topic[:30],
                    'Coverage Variance': topic_data['Percentage'].var(),
                    'Max Coverage': topic_data['Percentage'].max(),
                    'Min Coverage': topic_data['Percentage'].min(),
                    'Range': topic_data['Percentage'].max() - topic_data['Percentage'].min()
                })

        var_df = pd.DataFrame(variance_data).sort_values('Range', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Highest Variation (potential selection bias)**")
            st.dataframe(
                var_df.head(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
                use_container_width=True
            )
        with col2:
            st.markdown("**Most Consistent Coverage**")
            st.dataframe(
                var_df.tail(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
                use_container_width=True
            )


def render_sentiment_tab():
    """Render sentiment analysis tab with multi-model comparison."""
    st.subheader("😊 Sentiment Analysis - Multi-Model Comparison")

    # Load available data
    available_models = load_available_models()
    topics = load_topic_list()

    if not available_models:
        st.warning("No sentiment analysis data found. Run `python scripts/04_analyze_sentiment.py` first.")
        return

    # Sidebar: Global topic selector
    st.sidebar.markdown("### Filters")
    topic_options = ["All Topics"] + [t['name'] for t in topics]
    selected_topic = st.sidebar.selectbox(
        "Select Topic",
        options=topic_options,
        help="Filter all visualizations by topic"
    )

    # Show available models
    model_list = [m['model_type'] for m in available_models]
    st.sidebar.markdown("### Available Models")
    for m in available_models:
        st.sidebar.text(f"✓ {m['model_type']}: {m['article_count']:,} articles")

    # View mode selector
    view_mode = st.radio(
        "View Mode",
        options=["Single Model View", "Model Comparison View"],
        horizontal=True
    )

    st.divider()

    if view_mode == "Single Model View":
        render_single_model_view(model_list, selected_topic)
    else:
        render_model_comparison_view(model_list, selected_topic)


def render_single_model_view(available_models: list, selected_topic: str):
    """Render single model view with topic filtering."""

    # Model selector
    model_display_names = {
        'roberta': 'RoBERTa',
        'distilbert': 'DistilBERT',
        'finbert': 'FinBERT',
        'vader': 'VADER',
        'textblob': 'TextBlob',
        'local': 'Local (RoBERTa)'  # Backward compatibility
    }

    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        format_func=lambda x: model_display_names.get(x, x.upper())
    )

    st.markdown(f"### {model_display_names.get(selected_model, selected_model.upper())} Analysis")
    if selected_topic != "All Topics":
        st.caption(f"Filtered by topic: **{selected_topic}**")

    # Load data with topic filter
    sentiment_data = load_sentiment_by_source_topic(selected_model, selected_topic)

    if not sentiment_data:
        st.warning(f"No data for {selected_model} with topic '{selected_topic}'")
        return

    # Use existing single model rendering
    render_sentiment_single_model_charts(selected_model, selected_topic)


def render_model_comparison_view(available_models: list, selected_topic: str):
    """Render multi-model comparison view."""

    st.markdown("### Model Comparison")
    if selected_topic != "All Topics":
        st.caption(f"Filtered by topic: **{selected_topic}**")

    # Load multi-model data
    comparison_data = load_multi_model_comparison(available_models, selected_topic)

    if not comparison_data:
        st.warning("No comparison data available")
        return

    df = pd.DataFrame(comparison_data)
    df['source_name'] = df['source_id'].map(SOURCE_NAMES)

    # Color scheme for models
    MODEL_COLORS = {
        "roberta": "#1f77b4",
        "distilbert": "#ff7f0e",
        "finbert": "#2ca02c",
        "vader": "#d62728",
        "textblob": "#9467bd",
        "local": "#1f77b4"  # Backward compatibility
    }

    # 1. Multi-model stacked bar (grouped by source)
    st.markdown("#### Sentiment Distribution by Source & Model")
    st.caption("Percentage of negative/neutral/positive articles for each model, grouped by source")
    render_multi_model_stacked_bars(df, MODEL_COLORS)

    # 2. Average sentiment comparison (grouped bar chart)
    st.markdown("#### Average Sentiment: Source × Model Comparison")
    render_source_model_comparison(df, MODEL_COLORS)

    # 3. Model agreement analysis
    st.markdown("#### Model Agreement Matrix")
    st.caption("Correlation between models - higher values indicate models agree more")
    render_model_agreement_heatmap(df)


def render_multi_model_stacked_bars(df, model_colors):
    """Render stacked bars grouped by source, showing all models."""

    # Calculate percentages for each model
    results = []
    for source in df['source_name'].unique():
        for model in df['model_type'].unique():
            subset = df[(df['source_name'] == source) & (df['model_type'] == model)]

            if len(subset) == 0:
                continue

            total = len(subset)
            negative = len(subset[subset['overall_sentiment'] < -0.5])
            neutral = len(subset[(subset['overall_sentiment'] >= -0.5) &
                                 (subset['overall_sentiment'] <= 0.5)])
            positive = len(subset[subset['overall_sentiment'] > 0.5])

            results.append({
                'source': source,
                'model': model,
                'negative_pct': (negative / total) * 100,
                'neutral_pct': (neutral / total) * 100,
                'positive_pct': (positive / total) * 100
            })

    results_df = pd.DataFrame(results)

    # Create grouped bar chart (one per model)
    for model in sorted(df['model_type'].unique()):
        model_data = results_df[results_df['model'] == model]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Negative (< -0.5)',
            x=model_data['source'],
            y=model_data['negative_pct'],
            marker_color='#d62728',
            text=model_data['negative_pct'].round(1).astype(str) + '%',
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='Neutral (-0.5 to 0.5)',
            x=model_data['source'],
            y=model_data['neutral_pct'],
            marker_color='#7f7f7f',
            text=model_data['neutral_pct'].round(1).astype(str) + '%',
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='Positive (> 0.5)',
            x=model_data['source'],
            y=model_data['positive_pct'],
            marker_color='#2ca02c',
            text=model_data['positive_pct'].round(1).astype(str) + '%',
            textposition='inside'
        ))

        fig.update_layout(
            barmode='stack',
            height=300,
            title=f"{model.upper()} Model",
            xaxis_title="Source",
            yaxis_title="Percentage (%)",
            yaxis_range=[0, 100],
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)


def render_source_model_comparison(df, model_colors):
    """Grouped bar chart: avg sentiment by source for each model."""

    # Calculate average sentiment per source per model
    agg = df.groupby(['source_name', 'model_type'])['overall_sentiment'].mean().reset_index()

    fig = go.Figure()

    for model in sorted(df['model_type'].unique()):
        model_data = agg[agg['model_type'] == model]

        fig.add_trace(go.Bar(
            name=model.upper(),
            x=model_data['source_name'],
            y=model_data['overall_sentiment'],
            marker_color=model_colors.get(model, '#999')
        ))

    fig.update_layout(
        barmode='group',
        height=400,
        xaxis_title="News Source",
        yaxis_title="Average Sentiment (-5 to +5)",
        yaxis_range=[-5, 5]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")

    st.plotly_chart(fig, use_container_width=True)


def render_model_agreement_heatmap(df):
    """Create correlation heatmap showing model agreement."""

    # Pivot to get one column per model
    pivot = df.pivot_table(
        values='overall_sentiment',
        index=['source_id', 'topic'],
        columns='model_type',
        aggfunc='mean'
    )

    # Calculate correlation matrix
    corr = pivot.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Model",
        yaxis_title="Model"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_single_model_charts(model_type: str, selected_topic: str = None):
    """Render sentiment charts for a single model with optional topic filter."""

    # Check if data exists
    sentiment_data = load_sentiment_by_source_topic(model_type, selected_topic)
    if not sentiment_data:
        st.warning(f"No sentiment data found. Run `python scripts/04_analyze_sentiment.py` to analyze articles.")
        return

    # 1. Sentiment Distribution by Source (Stacked Bar Chart)
    st.markdown("#### Sentiment Distribution by Source")
    st.caption("Percentage of articles in each sentiment category")

    pct_data = load_sentiment_percentage_by_source_topic(model_type, selected_topic)
    if pct_data:
        pct_df = pd.DataFrame(pct_data)
        pct_df['source_name'] = pct_df['source_id'].map(SOURCE_NAMES)

        # Calculate percentages
        pct_df['negative_pct'] = (pct_df['negative_count'] / pct_df['total_count'] * 100)
        pct_df['neutral_pct'] = (pct_df['neutral_count'] / pct_df['total_count'] * 100)
        pct_df['positive_pct'] = (pct_df['positive_count'] / pct_df['total_count'] * 100)

        # Create stacked bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Negative (< -0.5)',
            x=pct_df['source_name'],
            y=pct_df['negative_pct'],
            marker_color='#d62728',
            text=pct_df['negative_pct'].round(1).astype(str) + '%',
            textposition='inside',
            hovertemplate='%{x}<br>Negative: %{y:.1f}%<br>Count: ' + pct_df['negative_count'].astype(str) + '<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Neutral (-0.5 to 0.5)',
            x=pct_df['source_name'],
            y=pct_df['neutral_pct'],
            marker_color='#7f7f7f',
            text=pct_df['neutral_pct'].round(1).astype(str) + '%',
            textposition='inside',
            hovertemplate='%{x}<br>Neutral: %{y:.1f}%<br>Count: ' + pct_df['neutral_count'].astype(str) + '<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Positive (> 0.5)',
            x=pct_df['source_name'],
            y=pct_df['positive_pct'],
            marker_color='#2ca02c',
            text=pct_df['positive_pct'].round(1).astype(str) + '%',
            textposition='inside',
            hovertemplate='%{x}<br>Positive: %{y:.1f}%<br>Count: ' + pct_df['positive_count'].astype(str) + '<extra></extra>'
        ))

        fig.update_layout(
            barmode='stack',
            yaxis_title="Percentage (%)",
            xaxis_title="News Source",
            height=400,
            yaxis_range=[0, 100],
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. Average Sentiment by Source
    st.markdown("#### Average Sentiment by Source")
    source_df = pd.DataFrame(sentiment_data)
    source_df['source_name'] = source_df['source_id'].map(SOURCE_NAMES)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=source_df['source_name'],
        y=source_df['avg_sentiment'],
        error_y=dict(type='data', array=source_df['stddev_sentiment']),
        marker_color=[SOURCE_COLORS.get(name, '#999') for name in source_df['source_name']],
        text=source_df['avg_sentiment'].round(2),
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title="Average Sentiment Score (-5 to +5)",
        xaxis_title="News Source",
        height=400,
        yaxis_range=[-5, 5],
        hovermode='x unified'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Sentiment Distribution (Box Plot)
    st.markdown("#### Sentiment Distribution")
    dist_data = load_sentiment_distribution(model_type)
    if dist_data:
        dist_df = pd.DataFrame(dist_data)
        dist_df['source_name'] = dist_df['source_id'].map(SOURCE_NAMES)

        fig = go.Figure()
        for source in dist_df['source_name'].unique():
            source_data = dist_df[dist_df['source_name'] == source]
            fig.add_trace(go.Box(
                y=source_data['overall_sentiment'],
                name=source,
                marker_color=SOURCE_COLORS.get(source, '#999')
            ))
        fig.update_layout(
            yaxis_title="Sentiment Score (-5 to +5)",
            height=400,
            yaxis_range=[-5, 5],
            showlegend=True
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # 4. Sentiment Timeline
    st.markdown("#### Sentiment Over Time")
    timeline_data = load_sentiment_timeline(model_type)
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['source_name'] = timeline_df['source_id'].map(SOURCE_NAMES)

        fig = px.line(
            timeline_df,
            x='date',
            y='avg_sentiment',
            color='source_name',
            color_discrete_map=SOURCE_COLORS,
            labels={'avg_sentiment': 'Avg Sentiment', 'date': 'Date', 'source_name': 'Source'}
        )
        fig.update_layout(height=400, yaxis_range=[-5, 5])
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Topic-Sentiment Heatmap
    st.markdown("#### Topic Sentiment by Source")
    topic_sentiment = load_topic_sentiment(model_type)
    if topic_sentiment:
        ts_df = pd.DataFrame(topic_sentiment)
        ts_df['source_name'] = ts_df['source_id'].map(SOURCE_NAMES)

        # Pivot for heatmap
        pivot = ts_df.pivot_table(
            values='avg_sentiment',
            index='topic',
            columns='source_name',
            aggfunc='mean'
        )

        # Only show top 15 topics by total article count
        topic_counts = ts_df.groupby('topic')['article_count'].sum().sort_values(ascending=False)
        top_topics = topic_counts.head(15).index
        pivot = pivot.loc[pivot.index.isin(top_topics)]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            zmin=-3,
            zmax=3,
            colorbar=dict(title="Sentiment")
        ))
        fig.update_layout(
            height=600,
            xaxis_title="News Source",
            yaxis_title="Topic"
        )
        st.plotly_chart(fig, use_container_width=True)





if __name__ == "__main__":
    main()
