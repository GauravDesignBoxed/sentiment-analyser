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

    /* File upload styling */
    .file-upload-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
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

# --- Emoji Sentiment Mapping and Utilities (FIXED multi-emoji handling) ---

EMOJI_SENTIMENT_MAPPING = {
    # Very Positive
    '😍': 0.9, '🥰': 0.9, '🤩': 0.9, '🥳': 0.9, '❤️': 0.9, '❤': 0.9, '🎉': 0.9, '🏆': 0.9, '🥇': 0.9,
    # Positive
    '😊': 0.8, '😃': 0.8, '😄': 0.8, '😂': 0.8, '🤣': 0.8, '😘': 0.8, '🤗': 0.8, '🙌': 0.8,
    '💕': 0.8, '💖': 0.8, '💗': 0.8, '💘': 0.8, '💙': 0.8, '💚': 0.8, '💜': 0.8, '🧡': 0.8, '💛': 0.8,
    '🎊': 0.8, '🌟': 0.8, '🔥': 0.8, '🌈': 0.8, '🌞': 0.8, '💯': 0.8,
    # Moderately Positive
    '😁': 0.7, '😎': 0.7, '😇': 0.7, '👍': 0.7, '👏': 0.7, '💪': 0.7, '🤟': 0.7, '⭐': 0.7, '✨': 0.7,
    '🎁': 0.7, '💎': 0.7, '☀️': 0.7, '🤍': 0.7,
    # Slightly Positive
    '😆': 0.6, '😋': 0.6, '😜': 0.6, '😝': 0.6, '😛': 0.6, '😉': 0.6, '😌': 0.6, '👌': 0.6,
    '🤞': 0.6, '✌️': 0.6, '🤘': 0.6, '🎈': 0.6, '😗': 0.6, '😙': 0.6, '😚': 0.6,
    # Mildly Positive
    '🙂': 0.5, '👊': 0.5, '✊': 0.5,
    # Neutral/Mild
    '🤤': 0.4, '🥺': 0.4, '🤭': 0.4, '😏': 0.3, '🤫': 0.3, '🤔': 0.2, '🙃': 0.2, '🤪': 0.3, '🤓': 0.3, '🧐': 0.1,
    # True Neutral
    '😐': 0.0, '😶': 0.0, '🤷': 0.0, '🤷‍♀️': 0.0, '🤷‍♂️': 0.0, '💭': 0.0, '👽': 0.0, '👾': 0.0, '🤖': 0.0,
    # Slightly Negative
    '😴': -0.2, '💤': -0.1, '🤐': -0.2, '🤨': -0.2, '😬': -0.3, '🙄': -0.4, '😒': -0.5, '😑': -0.3, '😷': -0.3, '🥲': -0.3,
    # Moderately Negative
    '😕': -0.4, '🙁': -0.5, '☹️': -0.5, '😞': -0.6, '😟': -0.5, '😔': -0.6, '😣': -0.5, '😖': -0.6, '😪': -0.4,
    '😓': -0.5, '🤧': -0.4, '🥵': -0.5, '🥶': -0.5,
    # Negative
    '😢': -0.7, '😥': -0.6, '😨': -0.7, '😰': -0.7, '😫': -0.7, '😩': -0.7, '😵': -0.7, '🤯': -0.6, '🤒': -0.6,
    '🤕': -0.6, '🤢': -0.7, '👎': -0.7,
    # Very Negative
    '😭': -0.8, '😤': -0.6, '😠': -0.8, '😈': -0.6, '👿': -0.8, '💀': -0.8, '☠️': -0.8, '🤮': -0.8, '💩': -0.8,
    # Extremely Negative
    '😡': -0.9, '🤬': -0.9, '💔': -0.9, '🖕': -0.9,
    # Context Dependent
    '👻': -0.3, '❣️': 0.7, '💋': 0.6
}

# Regex to find emoji runs; we will split into individual tokens
EMOJI_REGEX = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\u2600-\u26FF"          # Misc symbols
    "\u2700-\u27BF"          # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental symbols
    "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
    "]+", flags=re.UNICODE
)
FITZPATRICK_MOD_RE = re.compile(r'[\U0001F3FB-\U0001F3FF]')  # skin tones
ZWJ = '\u200D'
VS16 = '\uFE0F'

def split_emoji_run(run: str):
    out = []
    i = 0
    while i < len(run):
        ch = run[i]
        if i + 1 < len(run) and run[i + 1] == VS16:
            out.append(ch + VS16)
            i += 2
        else:
            out.append(ch)
            i += 1
    return out

def normalize_emoji(e: str):
    e = e.replace(VS16, '')
    e = FITZPATRICK_MOD_RE.sub('', e)
    e = e.replace(ZWJ, '')
    return e

def build_normalized_mapping(mapping: dict):
    norm = {}
    for k, v in mapping.items():
        nk = normalize_emoji(k)
        norm[nk] = v
    return norm

NORMALIZED_EMOJI_MAP = build_normalized_mapping(EMOJI_SENTIMENT_MAPPING)

def extract_emojis_display(text: str):
    if pd.isna(text) or text is None:
        return []
    runs = EMOJI_REGEX.findall(str(text))
    tokens = []
    for r in runs:
        tokens.extend(split_emoji_run(r))
    return tokens

def extract_emoji_tokens(text: str, normalize: bool = True):
    if pd.isna(text) or text is None:
        return []
    runs = EMOJI_REGEX.findall(str(text))
    tokens = []
    for r in runs:
        tokens.extend(split_emoji_run(r))
    if normalize:
        tokens = [normalize_emoji(t) for t in tokens if normalize_emoji(t)]
    return tokens

def analyze_emoji_sentiment(text):
    try:
        display_emojis = extract_emojis_display(text)
        norm_emojis = extract_emoji_tokens(text, normalize=True)
        if not norm_emojis:
            return 0.0, 0, "neutral", []

        scores = []
        pos_scores, neg_scores, neu_scores = [], [], []
        for e in norm_emojis:
            s = NORMALIZED_EMOJI_MAP.get(e, 0.0)
            scores.append(s)
            if s > 0.15:
                pos_scores.append(s)
            elif s < -0.15:
                neg_scores.append(s)
            else:
                neu_scores.append(s)

        emoji_count = len(norm_emojis)
        avg_score = float(np.mean(scores))
        max_score = max(scores)
        min_score = min(scores)

        pc, nc = len(pos_scores), len(neg_scores)
        if pc > nc and pc > 0:
            if avg_score > 0.05 or max_score > 0.3:
                sentiment = "positive"
            else:
                sentiment = "neutral"
        elif nc > pc and nc > 0:
            if avg_score < -0.05 or min_score < -0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        elif pc == nc and pc > 0:
            pos_int = np.mean(pos_scores) if pos_scores else 0
            neg_int = abs(np.mean(neg_scores)) if neg_scores else 0
            if pos_int > neg_int:
                sentiment = "positive"
            elif neg_int > pos_int:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            if avg_score > 0.05:
                sentiment = "positive"
            elif avg_score < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        return avg_score, emoji_count, sentiment, display_emojis
    except Exception:
        return 0.0, 0, "neutral", []

def get_emoji_statistics(df):
    stats = {
        'total_emojis': 0,
        'unique_emojis': set(),
        'emoji_frequency': {},
        'avg_emojis_per_text': 0,
        'texts_with_emojis': 0,
        'most_common_emoji': None,
        'emoji_sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
    }
    all_emojis = []
    tw = 0
    for text in df['original_text']:
        emos = extract_emoji_tokens(text, normalize=True)
        if emos:
            tw += 1
            all_emojis.extend(emos)
            stats['unique_emojis'].update(emos)
            for e in emos:
                stats['emoji_frequency'][e] = stats['emoji_frequency'].get(e, 0) + 1

    stats['total_emojis'] = len(all_emojis)
    stats['texts_with_emojis'] = tw
    stats['avg_emojis_per_text'] = len(all_emojis) / len(df) if len(df) > 0 else 0

    if stats['emoji_frequency']:
        stats['most_common_emoji'] = max(stats['emoji_frequency'].items(), key=lambda x: x[1])

    for emoji, count in stats['emoji_frequency'].items():
        sc = NORMALIZED_EMOJI_MAP.get(emoji, 0.0)
        if sc > 0.15:
            stats['emoji_sentiment_distribution']['positive'] += count
        elif sc < -0.15:
            stats['emoji_sentiment_distribution']['negative'] += count
        else:
            stats['emoji_sentiment_distribution']['neutral'] += count
    return stats

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
        st.markdown(
            '<div class="dashboard-header"><h1 class="dashboard-title">🔐 Secure Access Required</h1><p class="dashboard-subtitle">Please enter your credentials to access the dashboard</p></div>',
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", on_change=password_entered, key="username")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown(
            '<div class="dashboard-header"><h1 class="dashboard-title">🔐 Secure Access Required</h1><p class="dashboard-subtitle">Please enter your credentials to access the dashboard</p></div>',
            unsafe_allow_html=True
        )
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

# --- File Loading Function (NEW) ---
def load_data_file(uploaded_file):
    """Load data from uploaded CSV or Excel file"""
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
            return df, "CSV"

        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_name.endswith('.xlsx') else 'xlrd')
            return df, "Excel"

        else:
            st.error("❌ Unsupported file format. Please upload CSV or Excel files only.")
            return None, None

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None, None

# --- Utility Functions ---
def simple_tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def clean_text(text):
    """
    Clean text while preserving emojis.
    Avoid removing emoji characters (no blanket non-word removal).
    """
    if pd.isna(text) or text is None:
        return ""
    try:
        text = str(text)
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    except Exception:
        return ""

def remove_stopwords(text, custom_stopwords):
    if not text or text.strip() == "":
        return ""
    try:
        tokens = simple_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in custom_stopwords and len(token) > 2]
        return ' '.join(filtered_tokens)
    except Exception:
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
    except Exception:
        return 0.0, 0.0, "neutral"

# Cache VADER analyzer
_VADER = SentimentIntensityAnalyzer()

def analyze_vader(text):
    try:
        if not text or text.strip() == "":
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"
        scores = _VADER.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return scores, sentiment
    except Exception:
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"

def get_consensus_sentiment(textblob_sent, vader_sent, emoji_sent):
    """
    Enhanced consensus including emoji sentiment (majority vote).
    """
    sentiments = [textblob_sent, vader_sent, emoji_sent]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    if pos > neg and pos > neu:
        return "positive"
    if neg > pos and neg > neu:
        return "negative"
    if neu > pos and neu > neg:
        return "neutral"
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
        except Exception:
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
    if not words:
        return None
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
        <p class="dashboard-subtitle">Professional sentiment analysis with interactive visualizations powered by TextBlob, VADER, and Enhanced Emoji Intelligence</p>
    </div>
    ''', unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")

        uploaded_file = st.file_uploader(
            "📄 Upload Data File",
            type=['csv', 'xlsx', 'xls'],
            help="Select your CSV or Excel file for analysis"
        )

        st.markdown("""
        <div class="file-upload-info">
            <strong>📋 Supported Formats:</strong><br>
            • CSV files (.csv)<br>
            • Excel files (.xlsx, .xls)<br>
            • UTF-8, Latin-1, CP1252 encodings
        </div>
        """, unsafe_allow_html=True)

        stopwords_file = st.file_uploader(
            "📝 Custom Stopwords (Optional)",
            type=['txt'],
            help="Upload custom stopwords file"
        )

        use_default_stopwords = st.checkbox("✅ Use Built-in Stopwords", value=True)

        st.markdown("---")
        st.markdown("### 📊 Dashboard Features")
        st.markdown("""
        - Multi-format Support (CSV/Excel)
        - TextBlob & VADER Sentiment
        - Enhanced Emoji Intelligence (multi-emoji + normalization)
        - Real-time Analysis & Interactive Charts
        - KPI Metrics, Category Filtering, Export Options
        """)

    if uploaded_file is not None:
        try:
            df, file_type = load_data_file(uploaded_file)

            if df is not None:
                # Display file information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"✅ {file_type} file loaded successfully!")
                with col2:
                    st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")
                with col3:
                    st.info(f"📁 File: {uploaded_file.name}")

                with st.expander("👀 Data Preview", expanded=False):
                    st.dataframe(df.head(), use_container_width=True)
                    st.markdown(f"**Column Names:** {', '.join(df.columns.tolist())}")

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
                    status_text.text("🧹 Preprocessing text data...")
                    progress_bar.progress(15)
                    df['original_text'] = df[text_column].astype(str)
                    df['cleaned_text'] = df['original_text'].apply(clean_text)

                    status_text.text("🛑 Removing stopwords...")
                    progress_bar.progress(30)
                    df['processed_text'] = df['cleaned_text'].apply(lambda x: remove_stopwords(x, stopwords_set))

                    status_text.text("🎭 Analyzing emojis...")
                    progress_bar.progress(45)
                    emoji_results = df['original_text'].apply(analyze_emoji_sentiment)
                    df['emoji_score'] = [r[0] for r in emoji_results]
                    df['emoji_count'] = [r[1] for r in emoji_results]
                    df['emoji_sentiment'] = [r[2] for r in emoji_results]
                    df['emojis_found'] = [r[3] for r in emoji_results]

                    status_text.text("🔍 Running TextBlob analysis...")
                    progress_bar.progress(65)
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

                    status_text.text("🤝 Calculating consensus (Text + Emoji)...")
                    progress_bar.progress(90)
                    df['consensus_sentiment'] = df.apply(
                        lambda row: get_consensus_sentiment(
                            row['textblob_sentiment'], row['vader_sentiment'], row['emoji_sentiment']
                        ),
                        axis=1
                    )

                    # Calculate emoji statistics
                    emoji_stats = get_emoji_statistics(df)

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")

                    # Store results
                    st.session_state['dashboard_data'] = df
                    st.session_state['text_column'] = text_column
                    st.session_state['category_column'] = category_column
                    st.session_state['date_column'] = date_column
                    st.session_state['file_type'] = file_type
                    st.session_state['emoji_stats'] = emoji_stats

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
        file_type = st.session_state.get('file_type', 'Unknown')
        emoji_stats = st.session_state.get('emoji_stats', None)

        # Filters Section
        st.markdown("## 🔍 Dashboard Filters")
        filter_cols = st.columns(5)

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
            emoji_count_min = 0
            emoji_count_max = int(df['emoji_count'].max()) if 'emoji_count' in df.columns else 0
            emoji_count_range = st.slider(
                "Emoji Count",
                min_value=0,
                max_value=emoji_count_max if emoji_count_max > 0 else 10,
                value=(0, emoji_count_max if emoji_count_max > 0 else 10),
                step=1
            )

        with filter_cols[4]:
            top_n_records = st.selectbox(
                "Records to Display",
                options=[100, 500, 1000, 5000, len(df)],
                index=2
            )

        # Apply filters
        filtered_df = df[
            (df['consensus_sentiment'].isin(sentiment_filter)) &
            (df['textblob_polarity'] >= polarity_range[0]) &
            (df['textblob_polarity'] <= polarity_range[1]) &
            (df['emoji_count'] >= emoji_count_range[0]) &
            (df['emoji_count'] <= emoji_count_range[1])
        ].head(top_n_records)

        if category_filter and category_column != "None":
            filtered_df = filtered_df[filtered_df[category_column].isin(category_filter)]

        # KPI Cards Section
        st.markdown("## 📊 Key Performance Indicators")
        sentiment_counts = filtered_df['consensus_sentiment'].value_counts()

        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.markdown(create_kpi_card("Total Records", len(filtered_df), "total"), unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(create_kpi_card("Positive Sentiment", sentiment_counts.get('positive', 0), "positive"),
                        unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(create_kpi_card("Negative Sentiment", sentiment_counts.get('negative', 0), "negative"),
                        unsafe_allow_html=True)
        with kpi_cols[3]:
            neutral_mixed = sentiment_counts.get('neutral', 0) + sentiment_counts.get('mixed', 0)
            st.markdown(create_kpi_card("Neutral/Mixed", neutral_mixed, "neutral"), unsafe_allow_html=True)
        with kpi_cols[4]:
            avg_emoji_score = filtered_df['emoji_score'].mean() if 'emoji_score' in filtered_df.columns else 0.0
            st.markdown(create_kpi_card("Avg Emoji Score", int(avg_emoji_score * 1000), "total"), unsafe_allow_html=True)

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

        # Optional Emoji Overview
        if emoji_stats:
            st.markdown("## 🎭 Emoji Overview")
            ecols = st.columns(5)
            with ecols[0]:
                st.metric("Total Emojis", emoji_stats['total_emojis'])
            with ecols[1]:
                st.metric("Unique Emojis", len(emoji_stats['unique_emojis']))
            with ecols[2]:
                st.metric("Texts with Emojis", emoji_stats['texts_with_emojis'])
            with ecols[3]:
                st.metric("Avg Emojis/Text", f"{emoji_stats['avg_emojis_per_text']:.2f}")
            with ecols[4]:
                mc = emoji_stats['most_common_emoji'][0] if emoji_stats['most_common_emoji'] else "N/A"
                st.metric("Most Used Emoji", mc)

        # Charts Section
        st.markdown("## 📈 Interactive Visualizations")
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
                fig_words = create_word_frequency_chart(filtered_df)
                if fig_words:
                    st.plotly_chart(fig_words, use_container_width=True)

        st.plotly_chart(create_polarity_distribution(filtered_df), use_container_width=True)

        if category_column != "None" and category_column in filtered_df.columns:
            fig_heatmap = create_category_sentiment_heatmap(filtered_df, category_column)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)

        # Data Table Section
        st.markdown("## 📋 Detailed Data View")
        display_columns = [text_column, 'consensus_sentiment', 'textblob_polarity', 'vader_compound',
                           'emoji_score', 'emoji_count']
        if category_column != "None":
            display_columns.insert(1, category_column)

        # Emojis in a readable form
        if 'emojis_found' in filtered_df.columns:
            filtered_df = filtered_df.copy()
            filtered_df['emojis_display'] = filtered_df['emojis_found'].apply(lambda x: ' '.join(x) if x else '')
            display_columns.append('emojis_display')

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
                f"sentiment_dashboard_data_{file_type.lower()}.csv",
                "text/csv",
                use_container_width=True
            )

        with export_cols[1]:
            summary_stats = pd.DataFrame({
                'Metric': ['Total Records', 'Positive', 'Negative', 'Neutral', 'Mixed',
                           'Avg Polarity', 'Avg Subjectivity', 'Avg VADER', 'Avg Emoji Score', 'Total Emojis'],
                'Value': [len(filtered_df), sentiment_counts.get('positive', 0),
                          sentiment_counts.get('negative', 0), sentiment_counts.get('neutral', 0),
                          sentiment_counts.get('mixed', 0), f"{filtered_df['textblob_polarity'].mean():.3f}",
                          f"{filtered_df['textblob_subjectivity'].mean():.3f}",
                          f"{filtered_df['vader_compound'].mean():.3f}",
                          f"{filtered_df['emoji_score'].mean():.3f}" if 'emoji_score' in filtered_df.columns else "0.000",
                          int(emoji_stats['total_emojis']) if emoji_stats else 0]
            })
            csv_stats = summary_stats.to_csv(index=False)
            st.download_button(
                "📈 Download Summary Stats",
                csv_stats,
                f"sentiment_summary_stats_{file_type.lower()}.csv",
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
                    f"sentiment_category_breakdown_{file_type.lower()}.csv",
                    "text/csv",
                    use_container_width=True
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>📊 Professional Sentiment Analytics Dashboard | Powered by Streamlit & Plotly</p>
        <p>Built with ❤️ for data-driven insights | Supports CSV & Excel formats | Enhanced Emoji Sentiment</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
