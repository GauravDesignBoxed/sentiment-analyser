import streamlit as st
import pandas as pd
import string
import sys
import os

# Set up proper NLTK data path before importing
import nltk
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Import other libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="CSV Sentiment Analyzer", 
    layout="wide",
    page_icon="🧠"
)

# --- NLTK Setup with Force Download ---
@st.cache_resource
def setup_nltk():
    """Setup NLTK with force download and proper error handling"""
    st.info("🔄 Setting up NLTK resources...")
    
    try:
        # Force download required resources
        nltk.download('punkt', quiet=True, force=True)
        nltk.download('stopwords', quiet=True, force=True) 
        nltk.download('vader_lexicon', quiet=True, force=True)
        
        # Test imports to ensure they work
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # Test functionality
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        stop_words = set(stopwords.words('english'))
        
        st.success("✅ NLTK setup complete!")
        return True, stop_words
        
    except Exception as e:
        st.error(f"❌ NLTK setup failed: {e}")
        # Return empty set as fallback
        return False, set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                          'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                          'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                          'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                          'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                          'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                          'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                          'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                          'with', 'through', 'during', 'before', 'after', 'above', 'below',
                          'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                          'further', 'then', 'once'])

# Initialize NLTK
nltk_success, default_stopwords = setup_nltk()

# --- Authentication ---
def check_password():
    """Returns True if the user has correct credentials."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            if (
                st.session_state["username"] == st.secrets["auth"]["username"]
                and st.session_state["password"] == st.secrets["auth"]["password"]
            ):
                st.session_state["password_correct"] = True
                # Clean up
                if "password" in st.session_state:
                    del st.session_state["password"]
                if "username" in st.session_state:
                    del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        except KeyError:
            st.error("❌ Authentication not configured. Please contact administrator.")
            st.session_state["password_correct"] = False

    # Main auth logic
    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 Authentication Required")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔐 Authentication Required") 
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect credentials")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- Utility Functions ---
def safe_tokenize(text):
    """Tokenize text with fallback if NLTK fails"""
    try:
        if nltk_success:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text.lower())
        else:
            # Simple fallback tokenizer
            import re
            return re.findall(r'\b\w+\b', text.lower())
    except:
        # Ultimate fallback
        return text.lower().split()

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or text is None:
        return ""
    
    try:
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        text = ''.join(char for char in text if not char.isdigit())
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

def remove_stopwords(text, stopwords_set):
    """Remove stopwords from text"""
    try:
        if not text or text.strip() == "":
            return ""
        
        words = safe_tokenize(text)
        filtered_words = [word for word in words if word not in stopwords_set and len(word) > 2]
        return ' '.join(filtered_words)
    
    except Exception as e:
        st.error(f"Error removing stopwords: {e}")
        return text

def analyze_textblob(text):
    """Analyze sentiment using TextBlob"""
    try:
        if not text or text.strip() == "":
            return 0.0, 0.0, "neutral"
        
        blob = TextBlob(text)
        polarity = float(blob.polarity)
        subjectivity = float(blob.subjectivity)
        
        # Determine sentiment
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative" 
        else:
            sentiment = "neutral"
            
        return polarity, subjectivity, sentiment
        
    except Exception as e:
        st.error(f"TextBlob error: {e}")
        return 0.0, 0.0, "neutral"

def analyze_vader(text):
    """Analyze sentiment using VADER"""
    try:
        if not text or text.strip() == "":
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"
        
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Determine sentiment
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return scores, sentiment
        
    except Exception as e:
        st.error(f"VADER error: {e}")
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, "neutral"

def get_consensus_sentiment(textblob_sent, vader_sent):
    """Get consensus between TextBlob and VADER"""
    if textblob_sent == vader_sent:
        return textblob_sent
    elif "neutral" in [textblob_sent, vader_sent]:
        return "neutral"
    else:
        return "mixed"

def load_stopwords(custom_content=None, use_default=True):
    """Load stopwords with fallback options"""
    stopwords_set = set()
    
    # Add default stopwords
    if use_default:
        stopwords_set.update(default_stopwords)
    
    # Add custom stopwords
    if custom_content:
        try:
            if ',' in custom_content:
                custom_words = [word.strip().lower() for word in custom_content.split(',')]
            else:
                custom_words = [word.strip().lower() for word in custom_content.split('\n')]
            
            stopwords_set.update([word for word in custom_words if word])
            st.info(f"✅ Added {len(custom_words)} custom stopwords")
        except Exception as e:
            st.error(f"Error loading custom stopwords: {e}")
    
    return stopwords_set

# --- Main App ---
def main():
    # Header
    st.title("🧠 CSV Sentiment Analyzer")
    st.markdown("""
    **Professional sentiment analysis using TextBlob and VADER**
    
    📊 **Features:**
    - Dual sentiment analysis engines
    - Custom stopwords support  
    - Category-based analysis
    - Interactive visualizations
    - Export results
    """)
    
    # File uploads
    st.header("📂 Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="CSV file containing text data for analysis"
        )
    
    with col2:
        stopwords_file = st.file_uploader(
            "Custom stopwords (optional)",
            type=['txt'], 
            help="Text file with stopwords (comma or line separated)"
        )
    
    # Options
    use_default_stopwords = st.checkbox("Include default English stopwords", value=True)
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Configuration
            st.header("⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox("Select text column", df.columns)
            
            with col2:
                category_options = ["None"] + list(df.columns)
                category_column = st.selectbox("Select category column (optional)", category_options)
            
            # Preview data
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head())
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Text column sample:** {df[text_column].iloc[0][:200]}...")
            
            # Load stopwords
            custom_stopwords_content = None
            if stopwords_file is not None:
                custom_stopwords_content = stopwords_file.read().decode('utf-8')
            
            stopwords_set = load_stopwords(custom_stopwords_content, use_default_stopwords)
            st.info(f"📝 Using {len(stopwords_set)} stopwords")
            
            # Run analysis
            if st.button("🚀 Run Analysis", type="primary"):
                
                # Validate
                if text_column not in df.columns:
                    st.error("Selected text column not found")
                    return
                
                # Check for valid text data
                valid_texts = df[text_column].dropna()
                if len(valid_texts) == 0:
                    st.error("No valid text data found")
                    return
                
                # Progress tracking
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # Step 1: Text cleaning
                    status.text("🧹 Cleaning text...")
                    progress_bar.progress(20)
                    
                    df['cleaned_text'] = df[text_column].apply(clean_text)
                    
                    # Step 2: Remove stopwords  
                    status.text("🛑 Removing stopwords...")
                    progress_bar.progress(40)
                    
                    df['processed_text'] = df['cleaned_text'].apply(
                        lambda x: remove_stopwords(x, stopwords_set)
                    )
                    
                    # Step 3: TextBlob analysis
                    status.text("🔍 TextBlob analysis...")
                    progress_bar.progress(60)
                    
                    textblob_results = df['processed_text'].apply(analyze_textblob)
                    df['textblob_polarity'] = [r[0] for r in textblob_results]
                    df['textblob_subjectivity'] = [r[1] for r in textblob_results]  
                    df['textblob_sentiment'] = [r[2] for r in textblob_results]
                    
                    # Step 4: VADER analysis
                    status.text("⚡ VADER analysis...")
                    progress_bar.progress(80)
                    
                    vader_results = df['processed_text'].apply(analyze_vader)
                    df['vader_compound'] = [r[0]['compound'] for r in vader_results]
                    df['vader_positive'] = [r[0]['pos'] for r in vader_results]
                    df['vader_neutral'] = [r[0]['neu'] for r in vader_results]
                    df['vader_negative'] = [r[0]['neg'] for r in vader_results]
                    df['vader_sentiment'] = [r[1] for r in vader_results]
                    
                    # Step 5: Consensus
                    status.text("🤝 Calculating consensus...")
                    progress_bar.progress(90)
                    
                    df['consensus_sentiment'] = df.apply(
                        lambda row: get_consensus_sentiment(
                            row['textblob_sentiment'], 
                            row['vader_sentiment']
                        ), axis=1
                    )
                    
                    progress_bar.progress(100)
                    status.text("✅ Analysis complete!")
                    
                    # Store in session state
                    st.session_state['results'] = df
                    st.session_state['text_col'] = text_column
                    st.session_state['category_col'] = category_column
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
                    return
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return

    # Display results
    if 'results' in st.session_state:
        df = st.session_state['results']
        text_column = st.session_state['text_col']
        category_column = st.session_state['category_col']
        
        st.header("📊 Results")
        
        # Filter by category
        display_df = df.copy()
        if category_column != "None" and category_column in df.columns:
            categories = df[category_column].dropna().unique()
            selected_categories = st.multiselect(
                "Filter by categories", 
                categories,
                default=list(categories)[:5] if len(categories) > 5 else list(categories)
            )
            if selected_categories:
                display_df = df[df[category_column].isin(selected_categories)]
        
        # Summary metrics
        st.subheader("📈 Summary")
        sentiment_counts = display_df['consensus_sentiment'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(display_df))
        with col2:
            st.metric("Positive", sentiment_counts.get('positive', 0))
        with col3:
            st.metric("Negative", sentiment_counts.get('negative', 0)) 
        with col4:
            st.metric("Neutral/Mixed", 
                     sentiment_counts.get('neutral', 0) + sentiment_counts.get('mixed', 0))
        
        # Sample results
        st.subheader("🔍 Sample Results")
        sample_cols = [text_column, 'textblob_sentiment', 'vader_sentiment', 'consensus_sentiment']
        if category_column != "None":
            sample_cols.insert(1, category_column)
        
        st.dataframe(display_df[sample_cols].head(10), use_container_width=True)
        
        # Visualizations
        st.header("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment pie chart
            st.subheader("🥧 Sentiment Distribution")
            if len(sentiment_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
                sentiment_counts.plot.pie(
                    ax=ax, 
                    autopct='%1.1f%%',
                    colors=colors[:len(sentiment_counts)]
                )
                ax.set_ylabel('')
                st.pyplot(fig)
                plt.close()
        
        with col2:
            # Word cloud
            st.subheader("☁️ Word Cloud")
            try:
                text_for_cloud = ' '.join(display_df['processed_text'].dropna())
                if text_for_cloud.strip():
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        max_words=50
                    ).generate(text_for_cloud)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No text available for word cloud")
            except Exception as e:
                st.error(f"Word cloud error: {e}")
        
        # Category analysis
        if category_column != "None" and category_column in display_df.columns:
            st.subheader("📈 Sentiment by Category")
            category_sentiment = pd.crosstab(
                display_df[category_column], 
                display_df['consensus_sentiment']
            )
            st.bar_chart(category_sentiment)
        
        # Download results
        st.header("📥 Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            csv_full = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results",
                csv_full,
                "sentiment_analysis_full.csv",
                "text/csv"
            )
        
        with col2:
            # Summary only
            summary_cols = [text_column, 'consensus_sentiment', 'textblob_polarity', 'vader_compound']
            if category_column != "None":
                summary_cols.insert(1, category_column)
            
            csv_summary = display_df[summary_cols].to_csv(index=False)
            st.download_button(
                "📥 Download Summary",
                csv_summary, 
                "sentiment_analysis_summary.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
