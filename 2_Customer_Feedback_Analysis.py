
import streamlit as st

st.set_page_config(
    page_title="Customer Feedback Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("Customer Feedback Sentiment Analysis")
st.success("NLP page is rendering successfully!")

import transformers
from transformers import pipeline

import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Customer Feedback Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("Customer Feedback Sentiment Analysis")
st.write(
    "Analyze customer reviews using a pretrained "
    "Hugging Face sentiment model."
)

MODEL_NAME = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME
    )


st.info(
    "The model predicts positive, neutral, or negative "
    "sentiment. Predictions may not always be correct."
)

review = st.text_area(
    "Enter a customer review",
    placeholder="Example: The product quality was excellent...",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a customer review.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                model = load_sentiment_model()
                result = model(review, truncation=True)[0]

                sentiment = result["label"].lower()
                confidence = result["score"]

                st.subheader("Analysis Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Sentiment",
                        sentiment.title()
                    )

                with col2:
                    st.metric(
                        "Model Score",
                        f"{confidence:.2%}"
                    )

                st.caption(
                    "The model score is not a guarantee "
                    "of real-world accuracy."
                )

            except Exception as error:
                st.error(f"Unable to analyze review: {error}")
