"""Sri Lanka Media Bias Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import json
import html
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_db
from src.versions import (
    list_versions,
    get_version,
    create_version,
    find_version_by_config,
    get_default_topic_config,
    get_default_clustering_config,
    get_default_word_frequency_config,
    get_default_ner_config
)
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
def load_overview_stats(version_id=None):
    """Load overview statistics for a specific version."""
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

            if version_id:
                # Total topics for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.topics WHERE topic_id != -1 AND result_version_id = %s",
                    (version_id,)
                )
                total_topics = cur.fetchone()["count"]

                # Total clusters for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE result_version_id = %s",
                    (version_id,)
                )
                total_clusters = cur.fetchone()["count"]

                # Multi-source clusters for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE sources_count > 1 AND result_version_id = %s",
                    (version_id,)
                )
                multi_source = cur.fetchone()["count"]
            else:
                # Fallback for no version selected
                total_topics = 0
                total_clusters = 0
                multi_source = 0

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
def load_topics(version_id=None):
    """Load topic data for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT topic_id, name, description, article_count
                FROM {schema}.topics
                WHERE topic_id != -1 AND result_version_id = %s
                ORDER BY article_count DESC
            """, (version_id,))
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
def load_topic_by_source(version_id=None):
    """Load topic distribution by source for a specific version."""
    if not version_id:
        return []
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT t.name as topic, n.source_id, COUNT(*) as count
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                JOIN {schema}.news_articles n ON aa.article_id = n.id
                WHERE t.topic_id != -1
                  AND aa.result_version_id = %s
                  AND t.result_version_id = %s
                GROUP BY t.name, n.source_id
                ORDER BY count DESC
            """, (version_id, version_id))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_top_events(version_id=None, limit=20):
    """Load top event clusters for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id, ec.cluster_name, ec.article_count, ec.sources_count,
                       ec.date_start, ec.date_end
                FROM {schema}.event_clusters ec
                WHERE ec.result_version_id = %s
                ORDER BY ec.article_count DESC
                LIMIT {limit}
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_event_details(event_id, version_id=None):
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


@st.cache_data(ttl=300)
def load_word_frequencies(version_id=None, limit=50):
    """Load word frequencies for a specific version."""
    if not version_id:
        return {}

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT source_id, word, frequency, tfidf_score, rank
                FROM {schema}.word_frequencies
                WHERE result_version_id = %s
                  AND rank <= %s
                ORDER BY source_id, rank
            """, (version_id, limit))
            rows = cur.fetchall()

            # Group by source
            result = {}
            for row in rows:
                source = row['source_id']
                if source not in result:
                    result[source] = []
                result[source].append(row)
            return result


@st.cache_resource
def load_bertopic_model(version_id=None):
    """Load the saved BERTopic model for a specific version.

    Tries to load from database first (for team collaboration),
    then falls back to filesystem for backward compatibility.
    """
    if not version_id:
        return None

    # Strategy 1: Try loading from database
    from src.versions import get_model_from_version
    import tempfile

    try:
        # Extract model from database to temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"bertopic_{version_id[:8]}_")
        model_path = get_model_from_version(version_id, temp_dir)

        if model_path:
            try:
                model = BERTopic.load(model_path)
                return model
            except Exception as e:
                st.warning(f"Model found in database but failed to load: {e}")
    except Exception as e:
        # Database loading failed, will try filesystem
        pass

    # Strategy 2: Fallback to filesystem (backward compatibility)
    model_path = Path(__file__).parent.parent / "models" / f"bertopic_model_{version_id[:8]}"
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / "models" / "bertopic_model"

    if model_path.exists():
        try:
            return BERTopic.load(str(model_path))
        except Exception as e:
            st.warning(f"Could not load BERTopic model from filesystem: {e}")
            return None

    # Model not found anywhere
    st.info("ℹ️ BERTopic model not found. Run the pipeline to generate visualizations.")
    return None


def render_version_selector(analysis_type):
    """Render version selector for a specific analysis type.

    Args:
        analysis_type: 'topics', 'clustering', or 'word_frequency'

    Returns:
        version_id of selected version or None
    """
    # Load versions for this analysis type
    versions = list_versions(analysis_type=analysis_type)

    if not versions:
        st.warning(f"No {analysis_type} versions found!")
        st.info(f"Create a {analysis_type} version using the button below to get started")
        return None

    # Version selector
    version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
        for v in versions
    }

    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    selected_label = st.selectbox(
        f"Select {display_name} Version",
        options=list(version_options.keys()),
        index=0,
        key=f"{analysis_type}_version_selector"
    )

    version_id = version_options[selected_label]
    version = get_version(version_id)

    # Display version info in an expander
    with st.expander("ℹ️ Version Details"):
        st.markdown(f"**Name:** {version['name']}")
        if version['description']:
            st.markdown(f"**Description:** {version['description']}")
        st.markdown(f"**Created:** {version['created_at'].strftime('%Y-%m-%d %H:%M')}")

        # Pipeline status
        status = version['pipeline_status']
        st.markdown("**Pipeline Status:**")

        if analysis_type == 'word_frequency':
            # Word frequency only has one pipeline step
            st.caption(f"{'✅' if status.get('word_frequency') else '⭕'} Word Frequency")
        else:
            # Topics and clustering have embeddings + analysis
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"{'✅' if status.get('embeddings') else '⭕'} Embeddings")
            with cols[1]:
                if analysis_type == 'topics':
                    st.caption(f"{'✅' if status.get('topics') else '⭕'} Topics")
                else:
                    st.caption(f"{'✅' if status.get('clustering') else '⭕'} Clustering")

        # Configuration preview
        config = version['configuration']
        st.markdown("**Configuration:**")

        if analysis_type == 'word_frequency':
            # Word frequency-specific settings
            wf_config = config.get('word_frequency', {})
            st.caption(f"Random Seed: {config.get('random_seed', 42)}")
            st.caption(f"Ranking Method: {wf_config.get('ranking_method', 'N/A')}")
            if wf_config.get('ranking_method') == 'tfidf':
                st.caption(f"TF-IDF Scope: {wf_config.get('tfidf_scope', 'N/A')}")
            st.caption(f"Top N Words: {wf_config.get('top_n_words', 'N/A')}")
            st.caption(f"Min Word Length: {wf_config.get('min_word_length', 'N/A')}")

            # Custom stopwords
            stopwords = wf_config.get('custom_stopwords', [])
            if stopwords:
                st.caption(f"Custom Stopwords: {', '.join(stopwords[:5])}{'...' if len(stopwords) > 5 else ''}")

        elif analysis_type == 'topics':
            # General settings
            st.caption(f"Random Seed: {config.get('random_seed', 42)}")
            st.caption(f"Embedding Model: {config.get('embeddings', {}).get('model', 'N/A')}")

            # Topic-specific settings
            topics_config = config.get('topics', {})
            st.caption(f"Min Topic Size: {topics_config.get('min_topic_size', 'N/A')}")
            st.caption(f"Diversity: {topics_config.get('diversity', 'N/A')}")

            # Stopwords
            stopwords = topics_config.get('stop_words', [])
            if stopwords:
                st.caption(f"Stop Words: {', '.join(stopwords)}")

            # Vectorizer parameters
            vectorizer_config = topics_config.get('vectorizer', {})
            if vectorizer_config:
                ngram_range = vectorizer_config.get('ngram_range', 'N/A')
                st.caption(f"N-gram Range: {ngram_range}")
                st.caption(f"Min DF: {vectorizer_config.get('min_df', 'N/A')}")

            # UMAP parameters
            umap_config = topics_config.get('umap', {})
            if umap_config:
                st.caption(f"UMAP n_neighbors: {umap_config.get('n_neighbors', 'N/A')}")
                st.caption(f"UMAP n_components: {umap_config.get('n_components', 'N/A')}")
                st.caption(f"UMAP min_dist: {umap_config.get('min_dist', 'N/A')}")
                st.caption(f"UMAP metric: {umap_config.get('metric', 'N/A')}")

            # HDBSCAN parameters
            hdbscan_config = topics_config.get('hdbscan', {})
            if hdbscan_config:
                st.caption(f"HDBSCAN min_cluster_size: {hdbscan_config.get('min_cluster_size', 'N/A')}")
                st.caption(f"HDBSCAN metric: {hdbscan_config.get('metric', 'N/A')}")
                st.caption(f"HDBSCAN cluster_selection_method: {hdbscan_config.get('cluster_selection_method', 'N/A')}")

        else:  # clustering
            # General settings
            st.caption(f"Random Seed: {config.get('random_seed', 42)}")
            st.caption(f"Embedding Model: {config.get('embeddings', {}).get('model', 'N/A')}")

            # Clustering-specific settings
            clustering_config = config.get('clustering', {})
            st.caption(f"Similarity Threshold: {clustering_config.get('similarity_threshold', 'N/A')}")
            st.caption(f"Time Window: {clustering_config.get('time_window_days', 'N/A')} days")
            st.caption(f"Min Cluster Size: {clustering_config.get('min_cluster_size', 'N/A')}")

    return version_id


def render_create_version_button(analysis_type):
    """Render button to create a new version for a specific analysis type.

    Args:
        analysis_type: 'topics', 'clustering', 'word_frequency', or 'ner'
    """
    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    if st.button(f"➕ Create New {display_name} Version", key=f"create_{analysis_type}_btn"):
        st.session_state[f'show_create_{analysis_type}'] = True

    # Show create dialog if requested
    if st.session_state.get(f'show_create_{analysis_type}', False):
        render_create_version_form(analysis_type)


def render_create_version_form(analysis_type):
    """Render form for creating a new version.

    Args:
        analysis_type: 'topics', 'clustering', 'word_frequency', or 'ner'
    """
    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    st.markdown("---")
    st.subheader(f"Create New {display_name} Version")

    with st.form(f"create_{analysis_type}_form"):
        name = st.text_input("Version Name", placeholder=f"e.g., baseline-{analysis_type}")
        description = st.text_area("Description (optional)", placeholder="What makes this version unique?")

        # Configuration editor
        st.markdown("**Configuration (JSON)**")
        if analysis_type == 'topics':
            default_config = get_default_topic_config()
        elif analysis_type == 'clustering':
            default_config = get_default_clustering_config()
        elif analysis_type == 'word_frequency':
            default_config = get_default_word_frequency_config()
        elif analysis_type == 'ner':
            default_config = get_default_ner_config()
        else:
            default_config = {}

        config_str = st.text_area(
            "Edit configuration",
            value=json.dumps(default_config, indent=2),
            height=300,
            key=f"{analysis_type}_config_editor"
        )

        col1, col2 = st.columns(2)

        with col1:
            submit = st.form_submit_button("Create Version")
        with col2:
            cancel = st.form_submit_button("Cancel")

        if cancel:
            st.session_state[f'show_create_{analysis_type}'] = False
            st.rerun()

        if submit:
            if not name:
                st.error("Version name is required")
            else:
                try:
                    # Parse configuration
                    config = json.loads(config_str)

                    # Check if config already exists for this analysis type
                    existing = find_version_by_config(config, analysis_type=analysis_type)
                    if existing:
                        st.warning(f"A {analysis_type} version with this configuration already exists: **{existing['name']}**")
                        st.info(f"Version ID: {existing['id']}")
                    else:
                        # Create version
                        version_id = create_version(name, description, config, analysis_type=analysis_type)
                        st.success(f"✅ Created {analysis_type} version: {name}")
                        st.info(f"Version ID: {version_id}")

                        # Show pipeline instructions
                        st.markdown("**Next steps:** Run the pipeline")
                        if analysis_type == 'word_frequency':
                            st.code(f"""# Compute word frequencies
python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}""")
                        elif analysis_type == 'ner':
                            st.code(f"""# Extract named entities
python3 scripts/ner/01_extract_entities.py --version-id {version_id}""")
                        else:
                            st.code(f"""# Generate embeddings
python3 scripts/{analysis_type}/01_generate_embeddings.py --version-id {version_id}

# Run analysis
python3 scripts/{analysis_type}/02_{'discover_topics' if analysis_type == 'topics' else 'cluster_events'}.py --version-id {version_id}""")

                        # Hide dialog
                        st.session_state[f'show_create_{analysis_type}'] = False

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON configuration: {e}")
                except Exception as e:
                    st.error(f"Error creating version: {e}")


def main():
    st.title("🇱🇰 Sri Lanka Media Bias Detector")
    st.markdown("Analyzing coverage patterns across Sri Lankan English newspapers")

    # Initialize session state for tabs
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Load overview stats (no version required for coverage)
    stats = load_overview_stats()

    # Overview metrics
    st.header("Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Articles", f"{stats['total_articles']:,}")
    with col2:
        if stats['date_range']['min_date']:
            st.caption(f"**Date range:** {stats['date_range']['min_date']} to {stats['date_range']['max_date']}")

    st.divider()

    # Tabs for different views
    tab_names = ["📊 Coverage", "🏷️ Topics", "📰 Events", "📝 Word Frequency", "👤 Named Entities", "⚖️ Source Comparison", "😊 Sentiment"]

    # Create buttons to switch tabs (2 rows for 7 tabs)
    # First row: 4 tabs
    cols_row1 = st.columns(4)
    for idx in range(4):
        with cols_row1[idx]:
            if st.button(tab_names[idx], key=f"tab_{idx}",
                        type="primary" if st.session_state.active_tab == idx else "secondary"):
                st.session_state.active_tab = idx

    # Second row: 3 tabs
    cols_row2 = st.columns(3)
    for idx in range(4, 7):
        with cols_row2[idx - 4]:
            if st.button(tab_names[idx], key=f"tab_{idx}",
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
        render_word_frequency_tab()
    elif st.session_state.active_tab == 4:
        render_ner_tab()
    elif st.session_state.active_tab == 5:
        render_comparison_tab()
    elif st.session_state.active_tab == 6:
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
    """Render topics analysis and source comparison tab."""
    st.subheader("📊 Topic Analysis")

    # Version selector at the top
    version_id = render_version_selector('topics')

    # Create version button
    render_create_version_button('topics')

    if not version_id:
        return

    st.markdown("---")

    topics = load_topics(version_id)
    if not topics:
        st.warning("No topics found for this version. Run topic discovery first.")
        st.code(f"""python3 scripts/topics/01_generate_embeddings.py --version-id {version_id}
python3 scripts/topics/02_discover_topics.py --version-id {version_id}""")
        return

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

    topic_source_data = load_topic_by_source(version_id)
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

    # Source comparison section
    st.divider()

    # Topic coverage comparison
    st.markdown("### Topic Focus by Source")
    st.markdown("What percentage of each source's coverage goes to each topic?")

    if topic_source_data:
        # Initialize session state for topic pagination
        if 'topic_focus_page' not in st.session_state:
            st.session_state.topic_focus_page = 0

        # Calculate percentages per source
        source_totals = ts_df.groupby('source_name')['count'].sum()

        # Get all topics (we'll paginate through them)
        topics_per_page = 10
        total_topics = len(topics)
        max_page = (total_topics - 1) // topics_per_page

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Previous", disabled=st.session_state.topic_focus_page == 0, key="prev_topics"):
                st.session_state.topic_focus_page = max(0, st.session_state.topic_focus_page - 1)
                st.rerun()
        with col2:
            st.caption(f"Showing topics {st.session_state.topic_focus_page * topics_per_page + 1}-{min((st.session_state.topic_focus_page + 1) * topics_per_page, total_topics)} of {total_topics}")
        with col3:
            if st.button("Next →", disabled=st.session_state.topic_focus_page >= max_page, key="next_topics"):
                st.session_state.topic_focus_page = min(max_page, st.session_state.topic_focus_page + 1)
                st.rerun()

        # Get topics for current page
        start_idx = st.session_state.topic_focus_page * topics_per_page
        end_idx = start_idx + topics_per_page
        top_topic_names_comparison = [t['name'] for t in topics[start_idx:end_idx]]

        comparison_data = []
        for source in SOURCE_NAMES.values():
            source_data = ts_df[ts_df['source_name'] == source]
            total = source_totals.get(source, 1)

            for topic in top_topic_names_comparison:
                topic_count = source_data[source_data['topic'] == topic]['count'].sum()
                comparison_data.append({
                    'Source': source,
                    'Topic': topic,
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
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            xaxis=dict(tickfont=dict(size=14))  # Increased font size from default (~12) to 14
        )
        st.plotly_chart(fig, use_container_width=True)

        # # Selection bias indicator
        # st.markdown("### Selection Bias Indicators")
        # st.markdown("Topics where sources significantly differ in coverage")

        # Calculate variance in coverage percentage across sources
        variance_data = []
        for topic in top_topic_names_comparison:
            topic_data = comp_df[comp_df['Topic'] == topic]
            if len(topic_data) > 1:
                variance_data.append({
                    'Topic': topic,
                    'Coverage Variance': topic_data['Percentage'].var(),
                    'Max Coverage': topic_data['Percentage'].max(),
                    'Min Coverage': topic_data['Percentage'].min(),
                    'Range': topic_data['Percentage'].max() - topic_data['Percentage'].min()
                })

        var_df = pd.DataFrame(variance_data).sort_values('Range', ascending=False)

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.markdown("**Highest Variation (potential selection bias)**")
        #     st.dataframe(
        #         var_df.head(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
        #         use_container_width=True
        #     )
        # with col2:
        #     st.markdown("**Most Consistent Coverage**")
        #     st.dataframe(
        #         var_df.tail(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
        #         use_container_width=True
        #     )

    # BERTopic Visualizations
    st.divider()
    st.subheader("Topic Model Visualizations")

    topic_model = load_bertopic_model(version_id)
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
                # Show top 20 topics
                top_topics_ids = [t['topic_id'] for t in topics[:20]]
                fig = topic_model.visualize_barchart(top_n_topics=20, topics=top_topics_ids)
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
        st.info("BERTopic model not found. Save the model during topic discovery.")


def render_events_tab():
    """Render events analysis tab."""
    st.subheader("📰 Event Clustering Analysis")

    # Version selector at the top
    version_id = render_version_selector('clustering')

    # Create version button
    render_create_version_button('clustering')

    if not version_id:
        return

    st.markdown("---")

    events = load_top_events(version_id, 30)
    if not events:
        st.warning("No event clusters found for this version. Run clustering first.")
        st.code(f"""python3 scripts/clustering/01_generate_embeddings.py --version-id {version_id}
python3 scripts/clustering/02_cluster_events.py --version-id {version_id}""")
        return

    events_df = pd.DataFrame(events)

    # Filter to multi-source events
    multi_source_events = events_df[events_df['sources_count'] > 1]

    # Event selector
    event_options = {
        f"{e['cluster_name']}... ({e['article_count']} articles, {e['sources_count']} sources)": e['id']
        for _, e in multi_source_events.iterrows()
    }

    selected_event_label = st.selectbox(
        "Select an event to explore",
        options=list(event_options.keys())
    )

    if selected_event_label:
        event_id = event_options[selected_event_label]
        articles = load_event_details(event_id, version_id)

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



def render_word_frequency_tab():
    """Render word frequency analysis tab."""
    st.subheader("📝 Word Frequency Analysis")

    # Version selector
    version_id = render_version_selector('word_frequency')

    # Create version button
    render_create_version_button('word_frequency')

    if not version_id:
        st.info("👆 Select or create a word frequency version to view analysis")
        return

    # st.markdown("---")

    # Get version details
    version = get_version(version_id)
    if not version:
        st.error("Version not found")
        return
    
    config = version['configuration']
    wf_config = config.get('word_frequency', {})

    # # Show version info
    # with st.expander("ℹ️ Version Configuration", expanded=False):
        
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("Ranking Method", wf_config.get('ranking_method', 'N/A').upper())
    #     with col2:
    #         if wf_config.get('ranking_method') == 'tfidf':
    #             st.metric("TF-IDF Scope", wf_config.get('tfidf_scope', 'N/A'))
    #         else:
    #             st.metric("Top Words", wf_config.get('top_n_words', 50))
    #     with col3:
    #         st.metric("Min Word Length", wf_config.get('min_word_length', 3))

    # Load word frequencies
    word_freqs = load_word_frequencies(version_id)

    if not word_freqs:
        st.warning("⚠️ No word frequencies found for this version. Run the pipeline first:")
        st.code(f"python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}")
        return

    # Get configuration for display
    ranking_method = wf_config.get('ranking_method', 'frequency')

    # Top words per source
    st.subheader("Top Words by Source")

    # Create 2x2 grid for sources
    sources = list(word_freqs.keys())
    num_sources = len(sources)

    if num_sources == 0:
        st.warning("No sources found")
        return

    # Create columns based on number of sources
    if num_sources <= 2:
        cols = st.columns(num_sources)
    else:
        # First row
        cols1 = st.columns(2)
        # Second row if needed
        if num_sources > 2:
            cols2 = st.columns(min(2, num_sources - 2))
            cols = list(cols1) + list(cols2)
        else:
            cols = cols1

    for idx, (source_id, words) in enumerate(word_freqs.items()):
        if idx >= len(cols):
            break

        source_name = SOURCE_NAMES.get(source_id, source_id)

        with cols[idx]:
            st.markdown(f"**{source_name}**")

            # Prepare data
            df = pd.DataFrame(words)

            # Determine value column
            if ranking_method == 'frequency':
                value_col = 'frequency'
                label = 'Frequency'
            else:
                value_col = 'tfidf_score'
                label = 'TF-IDF Score'

            # Bar chart (top 20)
            fig = px.bar(
                df.head(20),
                x=value_col,
                y='word',
                orientation='h',
                labels={value_col: label, 'word': 'Word'},
                color_discrete_sequence=[SOURCE_COLORS.get(source_name, '#1f77b4')]
            )
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # # Cross-source comparison
    # st.divider()
    # st.subheader("Word Comparison Across Sources")

    # # Find common words across sources
    # word_sets = {source_id: set([w['word'] for w in words[:50]]) for source_id, words in word_freqs.items()}

    # # Calculate overlaps
    # if len(word_sets) >= 2:
    #     source_ids = list(word_sets.keys())

    #     # Create comparison matrix
    #     st.markdown("**Top Word Overlap Between Sources**")

    #     overlap_data = []
    #     for i, source1 in enumerate(source_ids):
    #         for source2 in source_ids[i+1:]:
    #             common = word_sets[source1] & word_sets[source2]
    #             overlap_pct = len(common) / 50 * 100
    #             overlap_data.append({
    #                 'Source 1': SOURCE_NAMES.get(source1, source1),
    #                 'Source 2': SOURCE_NAMES.get(source2, source2),
    #                 'Common Words': len(common),
    #                 'Overlap %': f"{overlap_pct:.1f}%"
    #             })

    #     if overlap_data:
    #         overlap_df = pd.DataFrame(overlap_data)
    #         st.dataframe(overlap_df, use_container_width=True, hide_index=True)

    # # Show unique words per source
    # st.markdown("**Distinctive Words per Source** (words appearing in top 50 of only one source)")

    # # Find words unique to each source
    # all_words = set()
    # for words_set in word_sets.values():
    #     all_words.update(words_set)

    # unique_words = {}
    # for source_id, words_set in word_sets.items():
    #     # Words that appear in this source but not in any other source's top 50
    #     other_words = set()
    #     for other_id, other_set in word_sets.items():
    #         if other_id != source_id:
    #             other_words.update(other_set)

    #     unique = words_set - other_words
    #     if unique:
    #         unique_words[source_id] = unique

    # # Display unique words
    # if unique_words:
    #     unique_cols = st.columns(len(unique_words))
    #     for idx, (source_id, words) in enumerate(unique_words.items()):
    #         source_name = SOURCE_NAMES.get(source_id, source_id)
    #         with unique_cols[idx]:
    #             st.markdown(f"**{source_name}**")
    #             if words:
    #                 st.write(", ".join(sorted(list(words)[:15])))
    #             else:
    #                 st.write("(none)")
    # else:
    #     st.info("No distinctive words found - all sources share similar vocabulary in their top 50 words")


@st.cache_data(ttl=300)
def load_entity_statistics(version_id=None, entity_type=None, limit=100):
    """Load entity statistics for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        return db.get_entity_statistics(
            result_version_id=version_id,
            entity_type=entity_type,
            limit=limit
        )


def render_article_with_entities(content: str, entities: list) -> str:
    """
    Generate HTML with inline entity highlighting.

    Args:
        content: Article text content
        entities: List of entity dicts with entity_text, entity_type, start_char, end_char, confidence

    Returns:
        HTML string with highlighted entities
    """
    import html

    # Entity type color mapping (distinct light pastel colors for readability)
    # All 13 entity types from the NER model
    entity_colors = {
        'PERSON': '#E3F2FD',      # light blue
        'ORG': '#F3E5F5',         # light purple
        'ORGANIZATION': '#F3E5F5', # light purple (alias)
        'LOC': '#E8F5E9',         # light green
        'LOCATION': '#E8F5E9',    # light green (alias)
        'GPE': '#C8E6C9',         # medium green (geopolitical entity)
        'DATE': '#FFF3E0',        # light orange
        'TIME': '#FFE0B2',        # medium orange
        'EVENT': '#FCE4EC',       # light pink
        'FAC': '#E1BEE7',         # light violet (facilities)
        'PRODUCT': '#FFECB3',     # light yellow
        'PERCENT': '#B2DFDB',     # light teal
        'NORP': '#D1C4E9',        # light lavender (nationalities/religious/political groups)
        'MONEY': '#C5E1A5',       # light lime
        'LAW': '#FFCCBC',         # light coral
    }
    default_color = '#F5F5F5'  # light gray

    if not entities:
        return f'<div style="line-height: 1.8; font-size: 16px; white-space: pre-wrap;">{html.escape(content)}</div>'

    # Build HTML by processing content and inserting entity spans
    html_parts = []
    last_end = 0

    # Track positions we've already highlighted to avoid overlaps
    highlighted_ranges = []

    for entity in entities:
        start = entity['start_char']
        end = entity['end_char']

        # Skip if this entity overlaps with a previously highlighted one
        is_overlap = any(
            (start < prev_end and end > prev_start)
            for prev_start, prev_end in highlighted_ranges
        )
        if is_overlap:
            continue

        # Add text before this entity
        if start > last_end:
            html_parts.append(html.escape(content[last_end:start]))

        # Add highlighted entity
        entity_text = content[start:end]
        entity_type = entity['entity_type']
        confidence = entity.get('confidence', 0.0)
        color = entity_colors.get(entity_type, default_color)

        entity_html = (
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'border-radius: 3px; cursor: help; border: 1px solid #ccc;" '
            f'title="{entity_type} (confidence: {confidence:.2f})">'
            f'{html.escape(entity_text)}'
            f'</span>'
        )
        html_parts.append(entity_html)

        highlighted_ranges.append((start, end))
        last_end = end

    # Add remaining text after last entity
    if last_end < len(content):
        html_parts.append(html.escape(content[last_end:]))

    html_content = ''.join(html_parts)
    return f'<div style="line-height: 1.8; font-size: 16px; white-space: pre-wrap;">{html_content}</div>'


def render_comparison_tab():
    """Render source comparison tab."""
    st.subheader("⚖️ Source Comparison")
    st.info("Source comparison features are integrated into other tabs:")
    st.markdown("""
    - **Topics Tab**: View topic coverage distribution and selection bias across sources
    - **Events Tab**: Compare multi-source coverage of the same events
    - **Word Frequency Tab**: Compare distinctive vocabulary across sources
    - **Sentiment Tab**: Compare sentiment patterns across sources
    """)


def render_ner_tab():
    """Render Named Entity Recognition analysis tab."""
    st.subheader("👤 Named Entity Recognition")

    # Version selector
    version_id = render_version_selector('ner')

    # Create version button
    render_create_version_button('ner')

    if not version_id:
        st.info("👆 Select or create an NER version to view analysis")
        return

    # Get version details
    version = get_version(version_id)
    if not version:
        st.error("Version not found")
        return

    # Check if pipeline is complete
    if not version.get('is_complete'):
        st.warning("⚠️ Pipeline incomplete. Run the extraction script:")
        st.code(f"python3 scripts/ner/01_extract_entities.py --version-id {version_id}")
        return

    st.divider()

    # Load entity type distribution
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT entity_type, COUNT(*) as count
                FROM {schema}.named_entities
                WHERE result_version_id = %s
                GROUP BY entity_type
                ORDER BY count DESC
            """, (version_id,))
            entity_type_stats = cur.fetchall()

    if not entity_type_stats:
        st.info("No entities found. Run the extraction pipeline.")
        return

    # Entity type distribution chart
    st.subheader("Entity Distribution by Type")
    df_types = pd.DataFrame(entity_type_stats)
    fig = px.bar(
        df_types,
        x='entity_type',
        y='count',
        labels={'entity_type': 'Entity Type', 'count': 'Count'},
        color='entity_type'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Filter by entity type
    st.subheader("Top Entities by Source")

    entity_type_filter = st.selectbox(
        "Filter by Entity Type",
        options=["All"] + [row['entity_type'] for row in entity_type_stats],
        key="ner_entity_type_filter"
    )

    # Load entity statistics
    entity_filter = None if entity_type_filter == "All" else entity_type_filter
    entity_stats = load_entity_statistics(version_id, entity_type=entity_filter, limit=100)

    if not entity_stats:
        st.info(f"No entities found for type: {entity_type_filter}")
        return

    # Create dataframe
    df_entities = pd.DataFrame(entity_stats)
    df_entities['source_name'] = df_entities['source_id'].map(SOURCE_NAMES)

    # Pivot for heatmap
    pivot = df_entities.pivot_table(
        index='entity_text',
        columns='source_name',
        values='mention_count',
        fill_value=0
    )

    # Show top 20 entities
    top_entities = pivot.sum(axis=1).sort_values(ascending=False).head(20)
    pivot_top = pivot.loc[top_entities.index]

    # Heatmap
    fig = px.imshow(
        pivot_top,
        labels=dict(x="Source", y="Entity", color="Mentions"),
        title=f"Top 20 {entity_type_filter} Entities by Source",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed table
    st.subheader("Detailed Entity Statistics")

    # Format dataframe for display
    display_df = df_entities[['entity_text', 'entity_type', 'source_name', 'mention_count', 'article_count']].copy()
    display_df.columns = ['Entity', 'Type', 'Source', 'Mentions', 'Articles']
    display_df = display_df.sort_values('Mentions', ascending=False)

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    # Article Entity Viewer Section
    st.divider()
    st.subheader("Article Entity Viewer")
    st.markdown("View named entities in context for any article from the corpus")

    # URL input
    article_url = st.text_input(
        "Enter Article URL",
        placeholder="Enter article URL from the corpus",
        key="ner_article_url_input"
    )

    if article_url:
        with get_db() as db:
            # Fetch article by URL
            article = db.get_article_by_url(article_url)

            if not article:
                st.warning("⚠️ Article not found. Please ensure the URL is exactly as stored in the database.")
            else:
                # Fetch entities for this article
                entities = db.get_entities_for_article(article['id'], version_id)

                # Display article metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Title:** {article['title']}")
                with col2:
                    source_name = SOURCE_NAMES.get(article['source_id'], article['source_id'])
                    st.markdown(f"**Source:** {source_name}")
                with col3:
                    st.markdown(f"**Date:** {article['date_posted']}")

                if not entities:
                    st.info("ℹ️ No entities were extracted from this article.")
                else:
                    # IMPORTANT: NER extraction uses title + "\n\n" + content
                    # So we need to reconstruct the same text to get correct positions
                    full_text = f"{article['title']}\n\n{article['content']}"

                    # Entity summary
                    entity_type_counts = {}
                    for entity in entities:
                        entity_type = entity['entity_type']
                        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

                    entity_summary = ", ".join([f"{count} {etype}" for etype, count in entity_type_counts.items()])
                    st.markdown(f"**Found {len(entities)} entities:** {entity_summary}")

                    # Entity legend
                    st.markdown("**Entity Type Legend:**")
                    legend_html = """
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; font-size: 14px;">
                        <span style="background-color: #E3F2FD; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">PERSON</span>
                        <span style="background-color: #F3E5F5; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">ORG</span>
                        <span style="background-color: #E8F5E9; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">LOC</span>
                        <span style="background-color: #C8E6C9; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">GPE</span>
                        <span style="background-color: #FFF3E0; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">DATE</span>
                        <span style="background-color: #FFE0B2; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">TIME</span>
                        <span style="background-color: #FCE4EC; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">EVENT</span>
                        <span style="background-color: #E1BEE7; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">FAC</span>
                        <span style="background-color: #FFECB3; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">PRODUCT</span>
                        <span style="background-color: #B2DFDB; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">PERCENT</span>
                        <span style="background-color: #D1C4E9; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">NORP</span>
                        <span style="background-color: #C5E1A5; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">MONEY</span>
                        <span style="background-color: #FFCCBC; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">LAW</span>
                    </div>
                    """
                    st.markdown(legend_html, unsafe_allow_html=True)

                    # Render article with highlighted entities
                    st.markdown("**Article with Highlighted Entities:**")
                    st.markdown("*(Hover over highlighted text to see entity type and confidence)*")

                    # Truncate long articles for display (show first 5000 characters)
                    is_truncated = len(full_text) > 5000
                    display_text = full_text[:5000] if is_truncated else full_text

                    # Filter entities to only those within the display range
                    display_entities = [e for e in entities if e['start_char'] < 5000]

                    html_content = render_article_with_entities(display_text, display_entities)
                    st.markdown(html_content, unsafe_allow_html=True)

                    if is_truncated:
                        st.info("📝 Article truncated for display (showing first 5000 characters)")



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
