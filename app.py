import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for deployment
import matplotlib.pyplot as plt
import os
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="CSV Sentiment Analyzer", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# --- NLTK Downloads with caching and error handling ---
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    downloads = ['punkt', 'stopwords', 'vader_lexicon']
    
    for item in downloads:
        try:
            # Check if already downloaded
            if item == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif item == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif item == 'vader_lexicon':
                nltk.data.find('vader_lexicon')
        except LookupError:
            try:
                nltk.download(item, quiet=True)
                st.success(f"✅ Downloaded {item}")
            except Exception as e:
                st.error(f"❌ Failed to download {item}: {e}")
                return False
    return True

# Initialize NLTK data
if not download_nltk_data():
    st.error("Failed to initialize NLTK data. Please refresh the page.")
    st.stop()

# --- Authentication Functions ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            if (
                st.session_state["username"] == st.secrets["auth"]["username"]
                and st.session_state["password"] == st.secrets["auth"]["password"]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store password
                del st.session_state["username"]  # Don't store username
            else:
                st.session_state["password_correct"] = False
        except KeyError:
            st.error("❌ Authentication configuration not found. Please contact administrator.")
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("## 🔐 Authentication Required")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.markdown("## 🔐 Authentication Required")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("❌ User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

# Check authentication
if not check_password():
    st.stop()

# --- Helper Functions ---
@st.cache_data
def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    try:
        text = str(text).lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove digits
        text = ''.join(char for char in text if not char.isdigit())
        return text.strip()
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

@st.cache_data
def remove_stopwords(text, custom_stopwords):
    """Remove stopwords from text"""
    try:
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in custom_stopwords]
        return ' '.join(filtered_words)
    except Exception as e:
        st.error(f"Error removing stopwords: {e}")
        return text

def analyze_textblob(text):
    """Analyze sentiment using TextBlob"""
    try:
        if not text or text.strip() == "":
            return 0.0, 0.0, "neutral"
        
        tb = TextBlob(text)
        polarity = tb.polarity
        subjectivity = tb.subjectivity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return polarity, subjectivity, sentiment
    except Exception as e:
        st.error(f"TextBlob analysis error: {e}")
        return 0.0, 0.0, "neutral"

def analyze_vader(text):
    """Analyze sentiment using VADER"""
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
        st.error(f"VADER analysis error: {e}")
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"

def get_consensus(tb_sent, vader_sent):
    """Get consensus sentiment from TextBlob and VADER"""
    if tb_sent == vader_sent:
        return tb_sent
    elif "neutral" in (tb_sent, vader_sent):
        return "neutral"
    else:
        return "mixed"

@st.cache_data
def load_custom_stopwords(stopword_content, use_default=True):
    """Load custom stopwords with default NLTK stopwords"""
    custom_stopwords = set()
    
    if use_default:
        try:
            custom_stopwords.update(stopwords.words("english"))
        except Exception as e:
            st.error(f"Error loading default stopwords: {e}")
    
    if stopword_content:
        try:
            if "," in stopword_content:
                # Comma-separated
                words = [word.strip().lower() for word in stopword_content.split(",")]
            else:
                # Line-separated
                words = [word.strip().lower() for word in stopword_content.splitlines()]
            
            custom_stopwords.update(words)
            st.info(f"✅ Loaded {len(words)} custom stopwords")
        except Exception as e:
            st.error(f"Error loading custom stopwords: {e}")
    
    return custom_stopwords

# --- Main Application ---
def main():
    # App Header
    st.title("🧠 CSV Text Sentiment Analyzer")
    st.markdown("""
    **Analyze sentiment in your CSV data using TextBlob and VADER sentiment analysis tools.**
    
    📊 Features:
    - Dual sentiment analysis (TextBlob + VADER)
    - Custom stopwords support
    - Category-based filtering
    - Interactive visualizations
    - Downloadable results
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Debug mode
        debug_mode = st.checkbox("🐛 Debug Mode", help="Show additional debugging information")
        
        if debug_mode:
            st.subheader("Debug Information")
            st.write("Python version:", sys.version)
            st.write("Streamlit version:", st.__version__)
            try:
                st.write("NLTK stopwords available:", len(stopwords.words("english")))
            except:
                st.write("NLTK stopwords: ❌ Not available")

    # File upload section
    st.header("📂 File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CSV file", 
            type=["csv"],
            help="Upload a CSV file containing text data for sentiment analysis"
        )
    
    with col2:
        stopword_file = st.file_uploader(
            "Optional: Upload custom stopwords (.txt)", 
            type=["txt"],
            help="Upload a text file with custom stopwords (comma or line separated)"
        )

    # Stopword options
    use_default_stopwords = st.checkbox(
        "Use NLTK English stopwords", 
        value=True,
        help="Include standard English stopwords from NLTK"
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Column selection
            st.header("🔧 Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "📝 Select the text column for analysis", 
                    df.columns,
                    help="Choose the column containing the text to analyze"
                )
            
            with col2:
                category_columns = ["None"] + list(df.columns)
                category_column = st.selectbox(
                    "📂 Optional: Select a category column", 
                    category_columns,
                    help="Choose a column to group results by category"
                )

            # Show data preview
            with st.expander("👀 Data Preview", expanded=False):
                st.dataframe(df.head())
                st.write(f"**Total rows:** {len(df)}")
                st.write(f"**Columns:** {list(df.columns)}")

            # Load stopwords
            stopword_content = ""
            if stopword_file is not None:
                stopword_content = stopword_file.read().decode("utf-8")
            
            custom_stopwords = load_custom_stopwords(stopword_content, use_default_stopwords)
            
            if custom_stopwords:
                st.info(f"📝 Total stopwords loaded: {len(custom_stopwords)}")

            # Analysis button
            if st.button("🚀 Run Sentiment Analysis", type="primary"):
                if text_column not in df.columns:
                    st.error("❌ Selected text column not found in data")
                    return
                
                # Check for empty data
                valid_rows = df[text_column].dropna()
                if len(valid_rows) == 0:
                    st.error("❌ No valid text data found in selected column")
                    return

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: Clean text
                    status_text.text("🧹 Cleaning text data...")
                    progress_bar.progress(20)
                    
                    df["cleaned_text"] = df[text_column].apply(clean_text)
                    
                    # Step 2: Remove stopwords
                    status_text.text("🛑 Removing stopwords...")
                    progress_bar.progress(40)
                    
                    df["no_stopwords"] = df["cleaned_text"].apply(
                        lambda x: remove_stopwords(x, custom_stopwords)
                    )
                    
                    # Step 3: TextBlob analysis
                    status_text.text("🔍 Running TextBlob analysis...")
                    progress_bar.progress(60)
                    
                    textblob_results = df["no_stopwords"].apply(analyze_textblob)
                    df["textblob_polarity"] = textblob_results.apply(lambda x: x[0])
                    df["textblob_subjectivity"] = textblob_results.apply(lambda x: x[1])
                    df["textblob_sentiment"] = textblob_results.apply(lambda x: x[2])
                    
                    # Step 4: VADER analysis
                    status_text.text("⚡ Running VADER analysis...")
                    progress_bar.progress(80)
                    
                    vader_results = df["no_stopwords"].apply(analyze_vader)
                    df["vader_compound"] = vader_results.apply(lambda x: x[0]['compound'])
                    df["vader_pos"] = vader_results.apply(lambda x: x[0]['pos'])
                    df["vader_neu"] = vader_results.apply(lambda x: x[0]['neu'])
                    df["vader_neg"] = vader_results.apply(lambda x: x[0]['neg'])
                    df["vader_sentiment"] = vader_results.apply(lambda x: x[1])
                    
                    # Step 5: Consensus
                    status_text.text("🤝 Calculating consensus...")
                    progress_bar.progress(90)
                    
                    df["consensus_sentiment"] = df.apply(
                        lambda row: get_consensus(row["textblob_sentiment"], row["vader_sentiment"]), 
                        axis=1
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Store results in session state
                    st.session_state["analysis_results"] = df
                    st.session_state["text_column"] = text_column
                    st.session_state["category_column"] = category_column
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    return

        except Exception as e:
            st.error(f"❌ Error loading CSV file: {e}")
            return

    # Display results if available
    if "analysis_results" in st.session_state:
        df = st.session_state["analysis_results"]
        text_column = st.session_state["text_column"]
        category_column = st.session_state["category_column"]
        
        st.header("📊 Analysis Results")
        
        # Category filtering
        if category_column != "None" and category_column in df.columns:
            st.subheader("🔍 Filter by Category")
            unique_categories = df[category_column].dropna().unique()
            selected_categories = st.multiselect(
                "Select categories to display",
                unique_categories,
                default=unique_categories[:5] if len(unique_categories) > 5 else unique_categories
            )
            
            if selected_categories:
                filtered_df = df[df[category_column].isin(selected_categories)]
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()

        # Results summary
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        sentiment_counts = filtered_df["consensus_sentiment"].value_counts()
        
        with col1:
            st.metric("Total Records", len(filtered_df))
        with col2:
            st.metric("Positive", sentiment_counts.get("positive", 0))
        with col3:
            st.metric("Negative", sentiment_counts.get("negative", 0))
        with col4:
            st.metric("Neutral/Mixed", 
                     sentiment_counts.get("neutral", 0) + sentiment_counts.get("mixed", 0))

        # Sample results
        st.subheader("🔍 Sample Results")
        display_columns = [
            text_column, 
            "textblob_sentiment", 
            "textblob_polarity",
            "vader_sentiment", 
            "vader_compound",
            "consensus_sentiment"
        ]
        
        if category_column != "None" and category_column in filtered_df.columns:
            display_columns.insert(1, category_column)
            
        st.dataframe(
            filtered_df[display_columns].head(10),
            use_container_width=True
        )

        # Visualizations
        st.header("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Sentiment Distribution")
            if len(sentiment_counts) > 0:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                colors = ["#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6"]
                sentiment_counts.plot.pie(
                    autopct='%1.1f%%', 
                    startangle=90, 
                    ax=ax1,
                    colors=colors[:len(sentiment_counts)]
                )
                ax1.set_ylabel("")
                ax1.set_title("Sentiment Distribution")
                st.pyplot(fig1)
                plt.close()
            else:
                st.info("No sentiment data to display")

        with col2:
            st.subheader("☁️ Word Cloud")
            try:
                all_text = " ".join(filtered_df["no_stopwords"].dropna())
                if all_text.strip():
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    ).generate(all_text)
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.imshow(wordcloud, interpolation='bilinear')
                    ax2.axis("off")
                    ax2.set_title("Most Common Words")
                    st.pyplot(fig2)
                    plt.close()
                else:
                    st.info("No text data available for word cloud")
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")

        # Category analysis
        if category_column != "None" and category_column in filtered_df.columns:
            st.subheader("📈 Sentiment by Category")
            
            category_sentiment = pd.crosstab(
                filtered_df[category_column], 
                filtered_df["consensus_sentiment"]
            )
            
            if not category_sentiment.empty:
                st.bar_chart(category_sentiment)
            else:
                st.info("No category data to display")

        # Detailed metrics
        with st.expander("📊 Detailed Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("TextBlob Metrics")
                st.write(f"**Average Polarity:** {filtered_df['textblob_polarity'].mean():.3f}")
                st.write(f"**Average Subjectivity:** {filtered_df['textblob_subjectivity'].mean():.3f}")
                
            with col2:
                st.subheader("VADER Metrics")
                st.write(f"**Average Compound:** {filtered_df['vader_compound'].mean():.3f}")
                st.write(f"**Average Positive:** {filtered_df['vader_pos'].mean():.3f}")
                st.write(f"**Average Negative:** {filtered_df['vader_neg'].mean():.3f}")

        # Download results
        st.header("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            csv_full = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Results",
                data=csv_full,
                file_name="sentiment_analysis_full_results.csv",
                mime="text/csv",
                help="Download complete analysis results with all columns"
            )
        
        with col2:
            # Summary results
            summary_columns = [
                text_column,
                "consensus_sentiment",
                "textblob_polarity",
                "vader_compound"
            ]
            if category_column != "None" and category_column in filtered_df.columns:
                summary_columns.insert(1, category_column)
                
            csv_summary = filtered_df[summary_columns].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Summary",
                data=csv_summary,
                file_name="sentiment_analysis_summary.csv",
                mime="text/csv",
                help="Download summary with key results only"
            )

# --- Run the application ---
if __name__ == "__main__":
    main()
