import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- NLTK Downloads ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# --- Basic Authentication ---
def login():
    st.title("🔐 Secure Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    if st.button("Login"):
        if (
            username_input == st.secrets["auth"]["username"]
            and password_input == st.secrets["auth"]["password"]
        ):
            st.session_state["authenticated"] = True
            st.success("✅ Logged in successfully!")
        else:
            st.error("❌ Incorrect username or password.")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

# --- App Title ---
st.set_page_config(page_title="CSV Sentiment Analyzer", layout="wide")
st.title("🧠 CSV Text Sentiment Analyzer (TextBlob + VADER)")
st.markdown("Upload a CSV file and get full sentiment analysis using TextBlob and VADER.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
stopword_file = st.file_uploader("🛑 Optional: Upload custom stopwords file (.txt)", type=["txt"])
use_default_stopwords = st.checkbox("Use NLTK English stopwords", value=True)

# --- Helper Functions ---
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(char for char in text if not char.isdigit())
    return text.strip()

def remove_stopwords(text, custom_stopwords):
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in custom_stopwords])

def analyze_textblob(text):
    tb = TextBlob(text)
    polarity = tb.polarity
    subjectivity = tb.subjectivity
    sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
    return polarity, subjectivity, sentiment

def analyze_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    sentiment = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
    return scores, sentiment

def get_consensus(tb_sent, vader_sent):
    if tb_sent == vader_sent:
        return tb_sent
    elif "neutral" in (tb_sent, vader_sent):
        return "neutral"
    return "mixed"

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    text_column = st.selectbox("📝 Select the text column", df.columns)
    category_column = st.selectbox("📂 Optional: Select a category column", ["None"] + list(df.columns))

    # --- Stopword Handling ---
    custom_stopwords = set()
    if use_default_stopwords:
        custom_stopwords.update(stopwords.words("english"))
    if stopword_file:
        content = stopword_file.read().decode("utf-8")
        if "," in content:
            custom_stopwords.update([word.strip().lower() for word in content.split(",")])
        else:
            custom_stopwords.update([word.strip().lower() for word in content.splitlines()])

    # --- Run Analysis ---
    if st.button("🚀 Run Sentiment Analysis"):
        with st.spinner("Analyzing... Please wait."):

            df["cleaned_text"] = df[text_column].apply(clean_text)
            df["no_stopwords"] = df["cleaned_text"].apply(lambda x: remove_stopwords(x, custom_stopwords))

            df["textblob_polarity"], df["textblob_subjectivity"], df["textblob_sentiment"] = zip(*df["no_stopwords"].map(analyze_textblob))
            vader_scores = df["no_stopwords"].map(analyze_vader)
            df["vader_compound"] = vader_scores.map(lambda x: x[0]['compound'])
            df["vader_sentiment"] = vader_scores.map(lambda x: x[1])
            df["consensus_sentiment"] = df.apply(
                lambda row: get_consensus(row["textblob_sentiment"], row["vader_sentiment"]), axis=1
            )

            st.success("✅ Analysis complete!")

            if category_column != "None":
                selected = st.multiselect("Filter by category", df[category_column].dropna().unique())
                if selected:
                    df = df[df[category_column].isin(selected)]

            st.subheader("🔍 Sample Output")
            st.dataframe(df[[text_column, "textblob_sentiment", "vader_sentiment", "consensus_sentiment"]].head())

            # --- Dashboard Style Charts ---
            st.subheader("📊 Sentiment Insights")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🥧 Sentiment Distribution Pie")
                sentiment_counts = df["consensus_sentiment"].value_counts()
                fig1, ax1 = plt.subplots()
                sentiment_counts.plot.pie(
                    autopct='%1.1f%%', startangle=90, ax=ax1,
                    colors=["#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6"]
                )
                ax1.set_ylabel("")
                st.pyplot(fig1)

            with col2:
                st.markdown("#### ☁️ Word Cloud")
                all_text = " ".join(df["no_stopwords"])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.imshow(wordcloud, interpolation='bilinear')
                ax2.axis("off")
                st.pyplot(fig2)

            col3, col4 = st.columns(2)
            with col4:
                if category_column != "None":
                    st.markdown("#### 📈 Sentiment by Category")
                    cat_sent = pd.crosstab(df[category_column], df["consensus_sentiment"])
                    st.bar_chart(cat_sent)

            st.subheader("📥 Download CSV")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, file_name="sentiment_output.csv", mime="text/csv")
