import streamlit as st
import pandas as pd
import string
import re
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

# --- Built-in English Stopwords (No NLTK needed) ---
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
        """Checks whether a password entered by the user is correct."""
        try:
            if (
                st.session_state["username"] == st.secrets["auth"]["username"]
                and st.session_state["password"] == st.secrets["auth"]["password"]
            ):
                st.session_state["password_correct"] = True
                # Clean up sensitive data
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        except KeyError:
            st.error("❌ Authentication not configured. Please contact administrator.")
            st.session_state["password_correct"] = False

    # Authentication flow
    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 Authentication Required")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔐 Authentication Required") 
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Invalid credentials")
        return False
    else:
        return True

# Check authentication
if not check_password():
    st.stop()

# --- Text Processing Functions (No NLTK) ---
def simple_tokenize(text):
    """Simple tokenizer using regex - no NLTK needed"""
    if not text:
        return []
    
    # Convert to lowercase and extract words
    text = text.lower()
    # Find all word characters, keeping contractions together
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def clean_text(text):
    """Clean and preprocess text without NLTK"""
    if pd.isna(text) or text is None:
        return ""
    
    try:
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation except apostrophes in contractions
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

def remove_stopwords(text, custom_stopwords):
    """Remove stopwords without NLTK"""
    if not text or text.strip() == "":
        return ""
    
    try:
        tokens = simple_tokenize(text)
        # Filter out stopwords and very short words
        filtered_tokens = [
            token for token in tokens 
            if token not in custom_stopwords and len(token) > 2
        ]
        return ' '.join(filtered_tokens)
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
        
        # Determine sentiment with thresholds
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
        
        # Determine sentiment
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

def get_consensus_sentiment(textblob_sent, vader_sent):
    """Calculate consensus between TextBlob and VADER"""
    if textblob_sent == vader_sent:
        return textblob_sent
    elif "neutral" in [textblob_sent, vader_sent]:
        return "neutral"
    else:
        return "mixed"

def load_stopwords(custom_content=None, use_default=True):
    """Load stopwords without NLTK dependency"""
    stopwords_set = set()
    
    # Add built-in English stopwords
    if use_default:
        stopwords_set.update(ENGLISH_STOPWORDS)
    
    # Add custom stopwords
    if custom_content:
        try:
            # Handle both comma and newline separated formats
            if ',' in custom_content:
                custom_words = [word.strip().lower() for word in custom_content.split(',')]
            else:
                custom_words = [word.strip().lower() for word in custom_content.split('\n')]
            
            # Filter empty strings
            custom_words = [word for word in custom_words if word]
            stopwords_set.update(custom_words)
            st.info(f"✅ Added {len(custom_words)} custom stopwords")
        except Exception as e:
            st.error(f"Error loading custom stopwords: {e}")
    
    return stopwords_set

# --- Main Application ---
def main():
    # App header
    st.title("🧠 CSV Sentiment Analyzer")
    st.markdown("""
    **Professional sentiment analysis using TextBlob and VADER** 
    
    🔹 **No NLTK dependencies** - Fully self-contained  
    🔹 **Dual sentiment engines** - TextBlob + VADER analysis  
    🔹 **Custom stopwords** - Upload your own stopwords  
    🔹 **Category filtering** - Group results by categories  
    🔹 **Visual insights** - Charts and word clouds  
    🔹 **Export results** - Download processed data  
    """)
    
    # File upload section
    st.header("📂 Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "📄 Upload CSV file", 
            type=['csv'],
            help="Select a CSV file containing text data to analyze"
        )
    
    with col2:
        stopwords_file = st.file_uploader(
            "📝 Upload custom stopwords (optional)",
            type=['txt'], 
            help="Text file with stopwords (comma-separated or line-separated)"
        )
    
    # Configuration options
    use_default_stopwords = st.checkbox(
        "✅ Use built-in English stopwords", 
        value=True,
        help="Include common English stopwords in filtering"
    )
    
    if uploaded_file is not None:
        try:
            # Load the CSV data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Column selection
            st.header("⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "📝 Select text column for analysis", 
                    df.columns,
                    help="Choose the column containing text to analyze"
                )
            
            with col2:
                category_options = ["None"] + list(df.columns)
                category_column = st.selectbox(
                    "📊 Select category column (optional)", 
                    category_options,
                    help="Choose a column to group results by category"
                )
            
            # Data preview
            with st.expander("👀 Preview Your Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show sample text
                if text_column in df.columns:
                    sample_text = str(df[text_column].iloc[0])[:300]
                    st.write(f"**Sample text:** {sample_text}...")
            
            # Load stopwords
            custom_stopwords_content = None
            if stopwords_file is not None:
                custom_stopwords_content = stopwords_file.read().decode('utf-8')
            
            stopwords_set = load_stopwords(custom_stopwords_content, use_default_stopwords)
            st.info(f"📝 Using {len(stopwords_set)} total stopwords")
            
            # Analysis button
            if st.button("🚀 Start Sentiment Analysis", type="primary"):
                
                # Validation
                if text_column not in df.columns:
                    st.error("❌ Selected text column not found in the data")
                    return
                
                # Check for valid text data
                valid_texts = df[text_column].dropna()
                if len(valid_texts) == 0:
                    st.error("❌ No valid text data found in selected column")
                    return
                
                # Progress tracking
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # Step 1: Text cleaning
                    status.text("🧹 Cleaning and preprocessing text...")
                    progress_bar.progress(20)
                    
                    df['cleaned_text'] = df[text_column].apply(clean_text)
                    
                    # Step 2: Remove stopwords
                    status.text("🛑 Removing stopwords...")
                    progress_bar.progress(40)
                    
                    df['processed_text'] = df['cleaned_text'].apply(
                        lambda x: remove_stopwords(x, stopwords_set)
                    )
                    
                    # Step 3: TextBlob sentiment analysis
                    status.text("🔍 Running TextBlob sentiment analysis...")
                    progress_bar.progress(60)
                    
                    textblob_results = df['processed_text'].apply(analyze_textblob)
                    df['textblob_polarity'] = [result[0] for result in textblob_results]
                    df['textblob_subjectivity'] = [result[1] for result in textblob_results]
                    df['textblob_sentiment'] = [result[2] for result in textblob_results]
                    
                    # Step 4: VADER sentiment analysis
                    status.text("⚡ Running VADER sentiment analysis...")
                    progress_bar.progress(80)
                    
                    vader_results = df['processed_text'].apply(analyze_vader)
                    df['vader_compound'] = [result[0]['compound'] for result in vader_results]
                    df['vader_positive'] = [result[0]['pos'] for result in vader_results]
                    df['vader_neutral'] = [result[0]['neu'] for result in vader_results]
                    df['vader_negative'] = [result[0]['neg'] for result in vader_results]
                    df['vader_sentiment'] = [result[1] for result in vader_results]
                    
                    # Step 5: Calculate consensus
                    status.text("🤝 Calculating consensus sentiment...")
                    progress_bar.progress(90)
                    
                    df['consensus_sentiment'] = df.apply(
                        lambda row: get_consensus_sentiment(
                            row['textblob_sentiment'], 
                            row['vader_sentiment']
                        ), axis=1
                    )
                    
                    # Complete
                    progress_bar.progress(100)
                    status.text("✅ Sentiment analysis completed successfully!")
                    
                    # Store results in session state
                    st.session_state['analysis_results'] = df
                    st.session_state['text_column'] = text_column
                    st.session_state['category_column'] = category_column
                    
                    st.balloons()  # Celebration!
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
                    return
                    
        except Exception as e:
            st.error(f"❌ Error loading CSV file: {str(e)}")
            return

    # Display results if analysis is complete
    if 'analysis_results' in st.session_state:
        df = st.session_state['analysis_results']
        text_column = st.session_state['text_column']
        category_column = st.session_state['category_column']
        
        st.header("📊 Analysis Results")
        
        # Category filtering
        display_df = df.copy()
        if category_column != "None" and category_column in df.columns:
            st.subheader("🔍 Filter Results")
            categories = df[category_column].dropna().unique()
            selected_categories = st.multiselect(
                "Select categories to display", 
                categories,
                default=list(categories)[:5] if len(categories) > 5 else list(categories),
                help="Choose which categories to include in the analysis"
            )
            if selected_categories:
                display_df = df[df[category_column].isin(selected_categories)]
        
        # Summary metrics
        st.subheader("📈 Summary Statistics")
        sentiment_counts = display_df['consensus_sentiment'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(display_df))
        with col2:
            st.metric("😊 Positive", sentiment_counts.get('positive', 0))
        with col3:
            st.metric("😞 Negative", sentiment_counts.get('negative', 0))
        with col4:
            neutral_mixed = sentiment_counts.get('neutral', 0) + sentiment_counts.get('mixed', 0)
            st.metric("😐 Neutral/Mixed", neutral_mixed)
        
        # Sample results table
        st.subheader("🔍 Sample Results")
        sample_columns = [text_column, 'textblob_sentiment', 'vader_sentiment', 'consensus_sentiment']
        if category_column != "None" and category_column in df.columns:
            sample_columns.insert(1, category_column)
        
        st.dataframe(
            display_df[sample_columns].head(10), 
            use_container_width=True
        )
        
        # Detailed metrics in expandable section
        with st.expander("📊 Detailed Sentiment Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**TextBlob Metrics:**")
                st.write(f"• Average Polarity: {display_df['textblob_polarity'].mean():.3f}")
                st.write(f"• Average Subjectivity: {display_df['textblob_subjectivity'].mean():.3f}")
                st.write(f"• Polarity Range: {display_df['textblob_polarity'].min():.3f} to {display_df['textblob_polarity'].max():.3f}")
            
            with col2:
                st.write("**VADER Metrics:**")
                st.write(f"• Average Compound: {display_df['vader_compound'].mean():.3f}")
                st.write(f"• Average Positive: {display_df['vader_positive'].mean():.3f}")
                st.write(f"• Average Negative: {display_df['vader_negative'].mean():.3f}")
        
        # Visualizations
        st.header("📊 Visual Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            st.subheader("🥧 Sentiment Distribution")
            if len(sentiment_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
                sentiment_counts.plot.pie(
                    ax=ax, 
                    autopct='%1.1f%%',
                    colors=colors[:len(sentiment_counts)],
                    startangle=90
                )
                ax.set_ylabel('')
                ax.set_title('Sentiment Distribution')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No sentiment data available for visualization")
        
        with col2:
            # Word cloud
            st.subheader("☁️ Most Common Words")
            try:
                text_for_cloud = ' '.join(display_df['processed_text'].dropna())
                if text_for_cloud.strip():
                    wordcloud = WordCloud(
                        width=800, 
                        height=400,
                        background_color='white',
                        max_words=50,
                        colormap='viridis'
                    ).generate(text_for_cloud)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud of Processed Text')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No text data available for word cloud")
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")
        
        # Category-based analysis
        if category_column != "None" and category_column in display_df.columns:
            st.subheader("📈 Sentiment by Category")
            category_sentiment = pd.crosstab(
                display_df[category_column], 
                display_df['consensus_sentiment']
            )
            
            if not category_sentiment.empty:
                st.bar_chart(category_sentiment)
            else:
                st.info("No category data available for visualization")
        
        # Download section
        st.header("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Full results download
            csv_full = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results",
                data=csv_full,
                file_name="sentiment_analysis_complete.csv",
                mime="text/csv",
                help="Download all columns including processed text and detailed scores"
            )
        
        with col2:
            # Summary download
            summary_columns = [text_column, 'consensus_sentiment', 'textblob_polarity', 'vader_compound']
            if category_column != "None" and category_column in df.columns:
                summary_columns.insert(1, category_column)
            
            csv_summary = display_df[summary_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download Summary",
                data=csv_summary,
                file_name="sentiment_analysis_summary.csv", 
                mime="text/csv",
                help="Download key results only"
            )
        
        with col3:
            # Statistics download
            stats_data = {
                'Metric': [
                    'Total Records',
                    'Positive Count',
                    'Negative Count', 
                    'Neutral Count',
                    'Mixed Count',
                    'Average TextBlob Polarity',
                    'Average TextBlob Subjectivity',
                    'Average VADER Compound'
                ],
                'Value': [
                    len(display_df),
                    sentiment_counts.get('positive', 0),
                    sentiment_counts.get('negative', 0),
                    sentiment_counts.get('neutral', 0),
                    sentiment_counts.get('mixed', 0),
                    f"{display_df['textblob_polarity'].mean():.3f}",
                    f"{display_df['textblob_subjectivity'].mean():.3f}",
                    f"{display_df['vader_compound'].mean():.3f}"
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics",
                data=csv_stats,
                file_name="sentiment_analysis_stats.csv",
                mime="text/csv",
                help="Download summary statistics"
            )

if __name__ == "__main__":
    main()
