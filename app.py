import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gradient_boosting_revenue.pkl"
FEATURES_PATH = BASE_DIR / "revenue_features.pkl"
NLP_PAGE_PATH = BASE_DIR / "2_Customer_Feedback_Analysis.py"

# --------------------------------------------------
# 2. LOAD MODEL AND FEATURES
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


# --------------------------------------------------
# 3. REVENUE PREDICTION PAGE
# --------------------------------------------------
def revenue_prediction_page():
    model, features = load_model()

    # Custom CSS
    st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 14px;
            background: linear-gradient(120deg, #17324d, #246b83);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.4rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.25);
            padding: 1rem;
            border-radius: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📊 Customer Intelligence")
        st.caption("Annual Revenue Prediction")
        st.divider()

        st.subheader("About this app")
        st.write(
            "Estimate annual customer revenue using a trained "
            "Gradient Boosting regression model."
        )

        st.info("Enter customer details, then click Predict Revenue.")
        st.caption("Model: Gradient Boosting")
        st.caption(f"Input features: {len(features)}")

    # Header
    st.markdown("""
    <div class="hero">
        <h1>Customer Intelligence Hub</h1>
        <p>Understand customer value with machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Enter a customer's profile below to estimate their annual revenue."
    )

    # Input form
    with st.form("customer_form"):
        st.subheader("Customer Profile")
        input_values = {}

        # Customer and engagement
        with st.expander("Customer & Engagement", expanded=True):
            cols = st.columns(3)

            for feature in [
                "Age",
                "Income",
                "Customer_Tenure_Months",
                "Website_Visits_Monthly",
                "App_Sessions_Monthly",
                "Purchase_Frequency_Monthly",
                "Orders_Last_12M",
                "Days_Since_Last_Purchase"
            ]:
                if feature in features:
                    with cols[len(input_values) % 3]:
                        if feature == "Age":
                            value = st.number_input(
                                feature,
                                min_value=18,
                                max_value=100,
                                value=30,
                                step=1
                            )
                        elif feature == "Income":
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                value=500000.0,
                                step=10000.0
                            )
                        else:
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                value=5.0,
                                step=1.0
                            )

                        input_values[feature] = value

        # Purchase behavior
        with st.expander("Purchase & Loyalty", expanded=True):
            cols = st.columns(3)

            for feature in [
                "Average_Order_Value",
                "Discount_Usage_Rate",
                "Return_Rate",
                "Loyalty_Points"
            ]:
                if feature in features:
                    with cols[len(input_values) % 3]:
                        if feature in [
                            "Discount_Usage_Rate",
                            "Return_Rate"
                        ]:
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                max_value=1.0,
                                value=0.1,
                                step=0.05
                            )
                        elif feature == "Average_Order_Value":
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                value=1500.0,
                                step=100.0
                            )
                        else:
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                value=100.0,
                                step=10.0
                            )

                        input_values[feature] = value

        # Satisfaction and service
        with st.expander("Satisfaction & Service", expanded=True):
            cols = st.columns(3)

            for feature in [
                "Customer_Satisfaction_Score",
                "Support_Tickets_12M",
                "Payment_Delay_Count",
                "Marketing_Response_Rate"
            ]:
                if feature in features:
                    with cols[len(input_values) % 3]:
                        if feature == "Customer_Satisfaction_Score":
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                max_value=5.0,
                                value=3.5,
                                step=0.5
                            )
                        elif feature == "Marketing_Response_Rate":
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                max_value=1.0,
                                value=0.2,
                                step=0.05
                            )
                        else:
                            value = st.number_input(
                                feature,
                                min_value=0.0,
                                value=1.0,
                                step=1.0
                            )

                        input_values[feature] = value

        submitted = st.form_submit_button(
            "📈 Predict Revenue",
            type="primary",
            use_container_width=True
        )

    # Prediction
    if submitted:
        try:
            input_df = pd.DataFrame(
                [input_values],
                columns=features
            )

            prediction = float(model.predict(input_df)[0])

            st.divider()
            st.subheader("Prediction Summary")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.metric(
                    "Estimated Annual Revenue",
                    f"₹{prediction:,.2f}"
                )

            with col2:
                st.metric("Model", "Gradient Boosting")

            st.success("Prediction generated successfully.")

            st.caption(
                "This is an estimate from the trained model, "
                "not a guarantee of actual customer revenue."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 4. APP NAVIGATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.stop()

if not FEATURES_PATH.exists():
    st.error(f"Feature file not found: {FEATURES_PATH.name}")
    st.stop()

if not NLP_PAGE_PATH.exists():
    st.error(f"NLP page file not found: {NLP_PAGE_PATH.name}")
    st.stop()

revenue_page = st.Page(
    revenue_prediction_page,
    title="Revenue Prediction",
    icon="📈",
    default=True
)

nlp_page = st.Page(
    str(NLP_PAGE_PATH),
    title="Customer Feedback Analysis",
    icon="💬"
)

navigation = st.navigation([revenue_page, nlp_page])
navigation.run()
