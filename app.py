import streamlit as st
import pandas as pd
import numpy as np
import string
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="📊 Sentiment Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Power BI Style ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container */
    .main > div {
        padding: 1rem 2rem;
    }
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .kpi-positive {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    
    .kpi-negative {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
    
    .kpi-neutral {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    }
    
    .kpi-total {
        background: linear-gradient(135deg, #2196F3 0%, #1976d2 100%);
    }
    
    .kpi-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2196F3;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Filter section */
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Built-in English Stopwords ---
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
    'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
    'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

# --- Authentication ---
def check_password():
    """Returns True if the user has correct credentials."""
    
    def password_entered():
        try:
            if (
                st.session_state["username"] == st.secrets["auth"]["username"]
                and st.session_state["password"] == st.secrets["auth"]["password"]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        except KeyError:
            st.error("❌ Authentication not configured. Please contact administrator.")
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown('<div class="dashboard-header"><h1 class="dashboard-title">🔐 Secure Access Required</h1><p class="dashboard-subtitle">Please enter your credentials to access the dashboard</p></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", on_change=password_entered, key="username")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown('<div class="dashboard-header"><h1 class="dashboard-title">🔐 Secure Access Required</h1><p class="dashboard-subtitle">Please enter your credentials to access the dashboard</p></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", on_change=password_entered, key="username")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            st.error("❌ Invalid credentials")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- Utility Functions ---
def simple_tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    try:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = ' '.join(text.split())
        return text.strip()
    except Exception as e:
        return ""

def remove_stopwords(text, custom_stopwords):
    if not text or text.strip() == "":
        return ""
    try:
        tokens = simple_tokenize(text)
        filtered_tokens = [
            token for token in tokens 
            if token not in custom_stopwords and len(token) > 2
        ]
        return ' '.join(filtered_tokens)
    except Exception as e:
        return text

def analyze_textblob(text):
    try:
        if not text or text.strip() == "":
            return 0.0, 0.0, "neutral"
        blob = TextBlob(text)
        polarity = float(blob.polarity)
        subjectivity = float(blob.subjectivity)
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return polarity, subjectivity, sentiment
    except Exception as e:
        return 0.0, 0.0, "neutral"

def analyze_vader(text):
    try:
        if not text or text.strip() == "":
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return scores, sentiment
    except Exception as e:
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"

def get_consensus_sentiment(textblob_sent, vader_sent):
    if textblob_sent == vader_sent:
        return textblob_sent
    elif "neutral" in [textblob_sent, vader_sent]:
        return "neutral"
    else:
        return "mixed"

def load_stopwords(custom_content=None, use_default=True):
    stopwords_set = set()
    if use_default:
        stopwords_set.update(ENGLISH_STOPWORDS)
    if custom_content:
        try:
            if ',' in custom_content:
                custom_words = [word.strip().lower() for word in custom_content.split(',')]
            else:
                custom_words = [word.strip().lower() for word in custom_content.split('\n')]
            custom_words = [word for word in custom_words if word]
            stopwords_set.update(custom_words)
        except Exception as e:
            pass
    return stopwords_set

# --- Power BI Style KPI Cards ---
def create_kpi_card(title, value, card_type="total"):
    kpi_class = f"kpi-card kpi-{card_type}"
    return f"""
    <div class="{kpi_class}">
        <p class="kpi-number">{value:,}</p>
        <p class="kpi-label">{title}</p>
    </div>
    """

# --- Advanced Visualizations ---
def create_sentiment_donut_chart(sentiment_counts):
    """Create a Power BI style donut chart"""
    colors = {
        'positive': '#4CAF50',
        'negative': '#f44336', 
        'neutral': '#ff9800',
        'mixed': '#9c27b0'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_counts.index),
        values=list(sentiment_counts.values),
        hole=0.6,
        marker=dict(colors=[colors.get(label, '#cccccc') for label in sentiment_counts.index]),
        textinfo='label+percent',
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '📊 Sentiment Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter'}
        },
        font=dict(family="Inter", size=12),
        height=400,
        margin=dict(t=80, b=40, l=40, r=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_sentiment_trend_chart(df, date_column=None):
    """Create sentiment trend over time"""
    if date_column and date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            daily_sentiment = df.groupby([df[date_column].dt.date, 'consensus_sentiment']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            colors = {'positive': '#4CAF50', 'negative': '#f44336', 'neutral': '#ff9800', 'mixed': '#9c27b0'}
            
            for sentiment in daily_sentiment.columns:
                fig.add_trace(go.Scatter(
                    x=daily_sentiment.index,
                    y=daily_sentiment[sentiment],
                    mode='lines+markers',
                    name=sentiment.title(),
                    line=dict(color=colors.get(sentiment, '#cccccc'), width=3),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title={'text': '📈 Sentiment Trend Over Time', 'x': 0.5, 'xanchor': 'center'},
                xaxis_title='Date',
                yaxis_title='Count',
                hovermode='x unified',
                height=400,
                font=dict(family="Inter"),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
        except:
            pass
    
    return None

def create_polarity_distribution(df):
    """Create polarity distribution histogram"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('TextBlob Polarity', 'VADER Compound'))
    
    fig.add_trace(go.Histogram(
        x=df['textblob_polarity'],
        nbinsx=30,
        name='TextBlob',
        marker_color='#2196F3',
        opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=df['vader_compound'],
        nbinsx=30,
        name='VADER',
        marker_color='#ff9800',
        opacity=0.7
    ), row=1, col=2)
    
    fig.update_layout(
        title={'text': '📊 Polarity Score Distributions', 'x': 0.5, 'xanchor': 'center'},
        height=400,
        showlegend=False,
        font=dict(family="Inter"),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_category_sentiment_heatmap(df, category_column):
    """Create category vs sentiment heatmap"""
    if category_column != "None" and category_column in df.columns:
        pivot_table = pd.crosstab(df[category_column], df['consensus_sentiment'], normalize='index') * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            text=np.round(pivot_table.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 12},
            hovertemplate='Category: %{y}<br>Sentiment: %{x}<br>Percentage: %{z:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title={'text': '🔥 Category vs Sentiment Heatmap (%)', 'x': 0.5, 'xanchor': 'center'},
            height=max(400, len(pivot_table.index) * 30),
            font=dict(family="Inter"),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    return None

def create_word_frequency_chart(df):
    """Create top words frequency chart"""
    all_text = ' '.join(df['processed_text'].dropna())
    words = all_text.split()
    word_freq = pd.Series(words).value_counts().head(15)
    
    fig = go.Figure(data=[go.Bar(
        x=word_freq.values,
        y=word_freq.index,
        orientation='h',
        marker_color='#2196F3',
        text=word_freq.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title={'text': '📝 Top 15 Most Frequent Words', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Frequency',
        height=500,
        font=dict(family="Inter"),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

# --- Main Dashboard ---
def main():
    # Dashboard Header
    st.markdown('''
    <div class="dashboard-header">
        <h1 class="dashboard-title">📊 Sentiment Analytics Dashboard</h1>
        <p class="dashboard-subtitle">Professional sentiment analysis with interactive visualizations powered by TextBlob & VADER</p>
    </div>
    ''', unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")
        
        uploaded_file = st.file_uploader(
            "📄 Upload CSV File", 
            type=['csv'],
            help="Select your data file for analysis"
        )
        
        stopwords_file = st.file_uploader(
            "📝 Custom Stopwords (Optional)",
            type=['txt'], 
            help="Upload custom stopwords file"
        )
        
        use_default_stopwords = st.checkbox("✅ Use Built-in Stopwords", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Features")
        st.markdown("""
        - **Real-time Analysis**
        - **Interactive Charts** 
        - **KPI Metrics**
        - **Category Filtering**
        - **Export Options**
        """)

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Configuration section
            st.markdown("## 🔧 Analysis Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_column = st.selectbox("📝 Text Column", df.columns)
            
            with col2:
                category_options = ["None"] + list(df.columns)
                category_column = st.selectbox("📊 Category Column", category_options)
            
            with col3:
                date_options = ["None"] + [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                date_column = st.selectbox("📅 Date Column (Optional)", date_options)
                if date_column == "None":
                    date_column = None
            
            # Load stopwords
            custom_stopwords_content = None
            if stopwords_file is not None:
                custom_stopwords_content = stopwords_file.read().decode('utf-8')
            
            stopwords_set = load_stopwords(custom_stopwords_content, use_default_stopwords)
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                
                # Progress tracking
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # Analysis steps
                    status_text.text("🧹 Preprocessing text data...")
                    progress_bar.progress(20)
                    df['cleaned_text'] = df[text_column].apply(clean_text)
                    
                    status_text.text("🛑 Removing stopwords...")
                    progress_bar.progress(40)
                    df['processed_text'] = df['cleaned_text'].apply(lambda x: remove_stopwords(x, stopwords_set))
                    
                    status_text.text("🔍 Running TextBlob analysis...")
                    progress_bar.progress(60)
                    textblob_results = df['processed_text'].apply(analyze_textblob)
                    df['textblob_polarity'] = [r[0] for r in textblob_results]
                    df['textblob_subjectivity'] = [r[1] for r in textblob_results]
                    df['textblob_sentiment'] = [r[2] for r in textblob_results]
                    
                    status_text.text("⚡ Running VADER analysis...")
                    progress_bar.progress(80)
                    vader_results = df['processed_text'].apply(analyze_vader)
                    df['vader_compound'] = [r[0]['compound'] for r in vader_results]
                    df['vader_positive'] = [r[0]['pos'] for r in vader_results]
                    df['vader_neutral'] = [r[0]['neu'] for r in vader_results]
                    df['vader_negative'] = [r[0]['neg'] for r in vader_results]
                    df['vader_sentiment'] = [r[1] for r in vader_results]
                    
                    status_text.text("🤝 Calculating consensus...")
                    progress_bar.progress(90)
                    df['consensus_sentiment'] = df.apply(
                        lambda row: get_consensus_sentiment(row['textblob_sentiment'], row['vader_sentiment']), axis=1
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")
                    
                    # Store results
                    st.session_state['dashboard_data'] = df
                    st.session_state['text_column'] = text_column
                    st.session_state['category_column'] = category_column
                    st.session_state['date_column'] = date_column
                    
                    # Clear progress indicators
                    progress_container.empty()
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    return
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return

    # Display Dashboard
    if 'dashboard_data' in st.session_state:
        df = st.session_state['dashboard_data']
        text_column = st.session_state['text_column']
        category_column = st.session_state['category_column']
        date_column = st.session_state['date_column']
        
        # Filters Section
        st.markdown("## 🔍 Dashboard Filters")
        
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            sentiment_filter = st.multiselect(
                "Sentiment Filter",
                options=['positive', 'negative', 'neutral', 'mixed'],
                default=['positive', 'negative', 'neutral', 'mixed']
            )
        
        with filter_cols[1]:
            if category_column != "None" and category_column in df.columns:
                category_options = list(df[category_column].unique())
                category_filter = st.multiselect(
                    "Category Filter",
                    options=category_options,
                    default=category_options[:10] if len(category_options) > 10 else category_options
                )
            else:
                category_filter = None
        
        with filter_cols[2]:
            polarity_range = st.slider(
                "Polarity Range",
                min_value=-1.0,
                max_value=1.0,
                value=(-1.0, 1.0),
                step=0.1
            )
        
        with filter_cols[3]:
            top_n_records = st.selectbox(
                "Records to Display",
                options=[100, 500, 1000, 5000, len(df)],
                index=2
            )
        
        # Apply filters
        filtered_df = df[
            (df['consensus_sentiment'].isin(sentiment_filter)) &
            (df['textblob_polarity'] >= polarity_range[0]) &
            (df['textblob_polarity'] <= polarity_range[1])
        ].head(top_n_records)
        
        if category_filter and category_column != "None":
            filtered_df = filtered_df[filtered_df[category_column].isin(category_filter)]
        
        # KPI Cards Section
        st.markdown("## 📊 Key Performance Indicators")
        
        sentiment_counts = filtered_df['consensus_sentiment'].value_counts()
        
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            st.markdown(create_kpi_card("Total Records", len(filtered_df), "total"), unsafe_allow_html=True)
        
        with kpi_cols[1]:
            st.markdown(create_kpi_card("Positive Sentiment", sentiment_counts.get('positive', 0), "positive"), unsafe_allow_html=True)
        
        with kpi_cols[2]:
            st.markdown(create_kpi_card("Negative Sentiment", sentiment_counts.get('negative', 0), "negative"), unsafe_allow_html=True)
        
        with kpi_cols[3]:
            neutral_mixed = sentiment_counts.get('neutral', 0) + sentiment_counts.get('mixed', 0)
            st.markdown(create_kpi_card("Neutral/Mixed", neutral_mixed, "neutral"), unsafe_allow_html=True)
        
        # Additional KPI metrics
        kpi_cols2 = st.columns(4)
        
        with kpi_cols2[0]:
            avg_polarity = filtered_df['textblob_polarity'].mean()
            st.metric("Avg Polarity", f"{avg_polarity:.3f}", delta=f"{avg_polarity:.3f}")
        
        with kpi_cols2[1]:
            avg_subjectivity = filtered_df['textblob_subjectivity'].mean()
            st.metric("Avg Subjectivity", f"{avg_subjectivity:.3f}")
        
        with kpi_cols2[2]:
            avg_vader = filtered_df['vader_compound'].mean()
            st.metric("Avg VADER Score", f"{avg_vader:.3f}", delta=f"{avg_vader:.3f}")
        
        with kpi_cols2[3]:
            positive_rate = (sentiment_counts.get('positive', 0) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            st.metric("Positive Rate", f"{positive_rate:.1f}%")
        
        # Charts Section
        st.markdown("## 📈 Interactive Visualizations")
        
        # Row 1: Donut Chart and Trend
        chart_cols1 = st.columns(2)
        
        with chart_cols1[0]:
            if len(sentiment_counts) > 0:
                fig_donut = create_sentiment_donut_chart(sentiment_counts)
                st.plotly_chart(fig_donut, use_container_width=True)
        
        with chart_cols1[1]:
            if date_column:
                fig_trend = create_sentiment_trend_chart(filtered_df, date_column)
                if fig_trend:
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                # Word frequency chart as alternative
                fig_words = create_word_frequency_chart(filtered_df)
                st.plotly_chart(fig_words, use_container_width=True)
        
        # Row 2: Polarity Distribution
        st.plotly_chart(create_polarity_distribution(filtered_df), use_container_width=True)
        
        # Row 3: Category Heatmap (if applicable)
        if category_column != "None" and category_column in filtered_df.columns:
            fig_heatmap = create_category_sentiment_heatmap(filtered_df, category_column)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Data Table Section
        st.markdown("## 📋 Detailed Data View")
        
        display_columns = [text_column, 'consensus_sentiment', 'textblob_polarity', 'vader_compound']
        if category_column != "None":
            display_columns.insert(1, category_column)
        
        st.dataframe(
            filtered_df[display_columns].head(50),
            use_container_width=True,
            height=400
        )
        
        # Export Section
        st.markdown("## 📥 Export Dashboard Data")
        
        export_cols = st.columns(3)
        
        with export_cols[0]:
            csv_full = filtered_df.to_csv(index=False)
            st.download_button(
                "📊 Download Full Dataset",
                csv_full,
                "sentiment_dashboard_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        with export_cols[1]:
            summary_stats = pd.DataFrame({
                'Metric': ['Total Records', 'Positive', 'Negative', 'Neutral', 'Mixed', 
                          'Avg Polarity', 'Avg Subjectivity', 'Avg VADER'],
                'Value': [len(filtered_df), sentiment_counts.get('positive', 0), 
                         sentiment_counts.get('negative', 0), sentiment_counts.get('neutral', 0),
                         sentiment_counts.get('mixed', 0), f"{filtered_df['textblob_polarity'].mean():.3f}",
                         f"{filtered_df['textblob_subjectivity'].mean():.3f}", f"{filtered_df['vader_compound'].mean():.3f}"]
            })
            csv_stats = summary_stats.to_csv(index=False)
            st.download_button(
                "📈 Download Summary Stats",
                csv_stats,
                "sentiment_summary_stats.csv",
                "text/csv",
                use_container_width=True
            )
        
        with export_cols[2]:
            if category_column != "None" and category_column in filtered_df.columns:
                category_breakdown = pd.crosstab(filtered_df[category_column], filtered_df['consensus_sentiment'])
                csv_breakdown = category_breakdown.to_csv()
                st.download_button(
                    "📊 Download Category Breakdown",
                    csv_breakdown,
                    "sentiment_category_breakdown.csv",
                    "text/csv",
                    use_container_width=True
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>📊 Professional Sentiment Analytics Dashboard | Powered by Streamlit & Plotly</p>
        <p>Built with ❤️ for data-driven insights</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
