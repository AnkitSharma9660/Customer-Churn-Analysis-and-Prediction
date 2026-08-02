import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Telecom Churn Intelligence System",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern premium look
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f3f4f6;
    }
    
    /* Header Banner */
    .header-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Risk Badges */
    .risk-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-moderate {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to safely load pickle models across sklearn versions
@st.cache_resource
def load_models():
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '_loss' or module.startswith('_loss'):
                import sklearn._loss as loss_mod
                if name == 'CyHalfBinomialLoss':
                    return getattr(loss_mod, 'HalfBinomialLoss')
                return getattr(loss_mod, name, None)
            return super().find_class(module, name)

    rf_path = "rf_churn_model.pkl"
    gb_path = "gb_churn_model.pkl"

    if not os.path.exists(rf_path) and os.path.exists(os.path.join("Customer-Churn-Analysis-and-Prediction-main", rf_path)):
        rf_path = os.path.join("Customer-Churn-Analysis-and-Prediction-main", rf_path)
    if not os.path.exists(gb_path) and os.path.exists(os.path.join("Customer-Churn-Analysis-and-Prediction-main", gb_path)):
        gb_path = os.path.join("Customer-Churn-Analysis-and-Prediction-main", gb_path)

    with open(rf_path, "rb") as f:
        rf_model = CustomUnpickler(f).load()

    with open(gb_path, "rb") as f:
        gb_model = CustomUnpickler(f).load()

    return rf_model, gb_model

try:
    rf_model, gb_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading prediction models: {e}")
    models_loaded = False

# Application Header
st.markdown("""
<div class="header-container">
    <div class="header-title">📞 Customer Churn Intelligence System</div>
    <div class="header-subtitle">Advanced Machine Learning Predictive Risk Analytics & Retention Optimization</div>
</div>
""", unsafe_allow_html=True)

# Navigation / Mode Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=70)
st.sidebar.title("Navigation & Controls")

app_mode = st.sidebar.radio(
    "Select Application Module:",
    ["🎯 Single Customer Predictor", "📊 Exploratory Diagrams & Analytics", "🤖 Model Benchmarks"]
)

# Preset Profile Loader
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Test Profiles")

if "preset_gender" not in st.session_state:
    st.session_state.preset_gender = 0
    st.session_state.preset_senior = 0
    st.session_state.preset_partner = 1
    st.session_state.preset_dependents = 0
    st.session_state.preset_tenure = 12
    st.session_state.preset_phone = 1
    st.session_state.preset_multiple = 0
    st.session_state.preset_internet = "Fiber optic"
    st.session_state.preset_security = 0
    st.session_state.preset_backup = 0
    st.session_state.preset_device = 0
    st.session_state.preset_support = 0
    st.session_state.preset_tv = 1
    st.session_state.preset_movies = 1
    st.session_state.preset_contract = "Month-to-month"
    st.session_state.preset_paperless = 1
    st.session_state.preset_payment = "Electronic check"
    st.session_state.preset_monthly = 85.0

col_p1, col_p2 = st.sidebar.columns(2)
if col_p1.button("🔴 High Risk Profile"):
    st.session_state.preset_tenure = 2
    st.session_state.preset_contract = "Month-to-month"
    st.session_state.preset_internet = "Fiber optic"
    st.session_state.preset_security = 0
    st.session_state.preset_support = 0
    st.session_state.preset_monthly = 95.0
    st.session_state.preset_payment = "Electronic check"

if col_p2.button("🟢 Low Risk Profile"):
    st.session_state.preset_tenure = 48
    st.session_state.preset_contract = "Two year"
    st.session_state.preset_internet = "DSL"
    st.session_state.preset_security = 1
    st.session_state.preset_support = 1
    st.session_state.preset_monthly = 55.0
    st.session_state.preset_payment = "Bank transfer (automatic)"


if app_mode == "🎯 Single Customer Predictor":
    
    st.subheader("📋 Customer Profile Input")
    
    # Input Tabs for clean layout
    tab_demo, tab_serv, tab_bill = st.tabs([
        "👤 Demographics & Account", 
        "📶 Telecom Services", 
        "💳 Billing & Contract Terms"
    ])
    
    with tab_demo:
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female (0)" if x==0 else "Male (1)", key="gender_select")
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="senior_select")
        with col2:
            partner = st.selectbox("Has Partner", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="partner_select")
            dependents = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="dep_select")
        with col3:
            tenure = st.slider("Customer Tenure (Months)", 0, 72, int(st.session_state.preset_tenure), key="tenure_slider")

    with tab_serv:
        col1, col2, col3 = st.columns(3)
        with col1:
            phone = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="phone_select")
            if phone == 1:
                multiple = st.selectbox("Multiple Lines", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="mult_select")
            else:
                multiple = 0
                st.info("Multiple Lines auto-set to No (0)")
                
        with col2:
            internet = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(st.session_state.preset_internet), key="internet_select")
            
            if internet == "No":
                dsl, fiber, no_internet = 0, 0, 1
                security, backup, device, support, tv, movies = 0, 0, 0, 0, 0, 0
                st.info("All internet add-on services auto-set to No (0)")
            else:
                dsl = 1 if internet == "DSL" else 0
                fiber = 1 if internet == "Fiber optic" else 0
                no_internet = 0
                
                security = st.selectbox("Online Security", [0, 1], index=st.session_state.preset_security, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="sec_select")
                backup = st.selectbox("Online Backup", [0, 1], index=st.session_state.preset_backup, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="bak_select")
                
        with col3:
            if internet != "No":
                device = st.selectbox("Device Protection", [0, 1], index=st.session_state.preset_device, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="dev_select")
                support = st.selectbox("Tech Support", [0, 1], index=st.session_state.preset_support, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="sup_select")
                tv = st.selectbox("Streaming TV", [0, 1], index=st.session_state.preset_tv, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="tv_select")
                movies = st.selectbox("Streaming Movies", [0, 1], index=st.session_state.preset_movies, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="mov_select")

    with tab_bill:
        col1, col2, col3 = st.columns(3)
        with col1:
            contract = st.selectbox("Contract Duration", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(st.session_state.preset_contract), key="contract_select")
            c_month = 1 if contract == "Month-to-month" else 0
            c_year = 1 if contract == "One year" else 0
            c_two = 1 if contract == "Two year" else 0

        with col2:
            payment = st.selectbox(
                "Payment Method",
                ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                index=["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"].index(st.session_state.preset_payment),
                key="payment_select"
            )
            p_bank = 1 if payment == "Bank transfer (automatic)" else 0
            p_card = 1 if payment == "Credit card (automatic)" else 0
            p_elec = 1 if payment == "Electronic check" else 0
            p_mail = 1 if payment == "Mailed check" else 0

        with col3:
            paperless = st.selectbox("Paperless Billing", [0, 1], index=st.session_state.preset_paperless, format_func=lambda x: "No (0)" if x==0 else "Yes (1)", key="paper_select")
            monthly = st.number_input("Monthly Charges ($)", 18.0, 150.0, float(st.session_state.preset_monthly), step=1.0, key="monthly_num")
            total = float(tenure * monthly)
            st.metric("Estimated Total Lifetime Charges", f"${total:,.2f}")

    # Feature Vector Construction
    features = np.array([[
        gender, senior, partner, dependents,
        tenure, phone, multiple,
        security, backup, device, support,
        tv, movies, paperless,
        monthly, total,
        dsl, fiber, no_internet,
        c_month, c_year, c_two,
        p_bank, p_card, p_elec, p_mail
    ]])

    st.markdown("---")
    
    predict_btn = st.button("🔮 Compute Churn Risk Assessment", type="primary", use_container_width=True)

    if predict_btn or "last_pred" in st.session_state:
        st.session_state.last_pred = True
        
        if models_loaded:
            rf_pred = rf_model.predict(features)[0]
            rf_prob = rf_model.predict_proba(features)[0][1]

            gb_pred = gb_model.predict(features)[0]
            gb_prob = gb_model.predict_proba(features)[0][1]

            ensemble_prob = (rf_prob + gb_prob) / 2.0
            
            st.markdown("### 📊 Prediction & Risk Diagnosis Summary")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Random Forest Classifier</div>
                    <div class="metric-val">{rf_prob*100:.1f}%</div>
                    <div>{"⚠️ Churn Risk" if rf_pred==1 else "✅ Likely Retained"}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Gradient Boosting Classifier</div>
                    <div class="metric-val">{gb_prob*100:.1f}%</div>
                    <div>{"⚠️ Churn Risk" if gb_pred==1 else "✅ Likely Retained"}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m3:
                if ensemble_prob >= 0.5:
                    status_html = '<span class="risk-high">🔴 HIGH CHURN DANGER</span>'
                elif ensemble_prob >= 0.3:
                    status_html = '<span class="risk-moderate">🟡 MODERATE RISK</span>'
                else:
                    status_html = '<span class="risk-low">🟢 LOW RISK / LOYAL</span>'
                    
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Consensus Churn Risk</div>
                    <div class="metric-val">{ensemble_prob*100:.1f}%</div>
                    <div>{status_html}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Plotly Gauge Chart for Probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ensemble_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Combined Ensemble Churn Probability (%)", 'font': {'size': 18, 'color': "#e2e8f0"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#f43f5e" if ensemble_prob >= 0.5 else ("#eab308" if ensemble_prob >= 0.3 else "#22c55e")},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.2)'},
                        {'range': [30, 50], 'color': 'rgba(234, 179, 8, 0.2)'},
                        {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f8fafc"},
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Key Risk Factor Identification & Actionable Retention Plan
            c_risk, c_recom = st.columns(2)
            
            with c_risk:
                st.markdown("#### 🚨 Identified Risk Drivers")
                risk_factors = []
                if c_month == 1:
                    risk_factors.append("• **Month-to-Month Contract**: High flexibility increases churn rate.")
                if tenure < 12:
                    risk_factors.append("• **Low Tenure (< 12 mos)**: Early customer stage exhibits highest risk.")
                if fiber == 1:
                    risk_factors.append("• **Fiber Optic Service**: Higher monthly cost with strong market competition.")
                if security == 0 or support == 0:
                    risk_factors.append("• **Missing Support Services**: Lack of Online Security / Tech Support correlates with customer dissatisfaction.")
                if p_elec == 1:
                    risk_factors.append("• **Electronic Check Payment**: Higher churn tendency compared to automated bank billing.")
                if monthly > 70:
                    risk_factors.append("• **High Monthly Charges (>$70)**: Price sensitivity increases churn potential.")

                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(rf)
                else:
                    st.success("No critical high-risk indicators detected for this customer profile!")

            with c_recom:
                st.markdown("#### 💡 Targeted Retention Strategy")
                if ensemble_prob >= 0.5:
                    st.warning("⚠️ **Immediate Action Recommended:**")
                    st.markdown("1. **Contract Upgrade Incentive**: Offer a 15% discount on an annual contract.")
                    st.markdown("2. **Service Bundle**: Provide 3 months free Tech Support and Online Security.")
                    st.markdown("3. **Proactive Outreach**: Initiate direct call from Customer Success Team.")
                elif ensemble_prob >= 0.3:
                    st.info("🟡 **Proactive Retention Suggestion:**")
                    st.markdown("1. Offer upgrade to automatic payment method with $5 monthly credit.")
                    st.markdown("2. Send tailored feature utilization tips to increase product adoption.")
                else:
                    st.success("🟢 **Customer Health Excellent:**")
                    st.markdown("1. Candidate for referral reward programs.")
                    st.markdown("2. Recommend cross-selling premium streaming or device protection packages.")

elif app_mode == "📊 Exploratory Diagrams & Analytics":
    st.subheader("📊 Exploratory Data Analysis & Notebook Diagrams")
    st.write("Visualizations extracted directly from model training notebook (`Notebook.ipynb`).")
    
    diagram_dir = "diagram"
    if not os.path.exists(diagram_dir) and os.path.exists(os.path.join("Customer-Churn-Analysis-and-Prediction-main", diagram_dir)):
        diagram_dir = os.path.join("Customer-Churn-Analysis-and-Prediction-main", diagram_dir)
        
    diagram_categories = {
        "🎯 Churn Distribution": [
            ("churn_bar_chart.png", "Overall Churn Count Bar Chart"),
            ("churn_pie_chart.png", "Overall Churn Percentage Pie Chart")
        ],
        "📋 Demographic & Contract Distributions": [
            ("contract_distribution_pie.png", "Contract Type Distribution"),
            ("gender_distribution_pie.png", "Gender Distribution"),
            ("payment_method_distribution_pie.png", "Payment Method Distribution"),
            ("customer_stage_distribution_pie.png", "Customer Stage Distribution"),
            ("paperless_billing_distribution_pie.png", "Paperless Billing Distribution"),
            ("internet_service_distribution_pie.png", "Internet Service Distribution")
        ],
        "📈 Feature Relationships & Churn Correlations": [
            ("churn_by_contract.png", "Churn by Contract Type"),
            ("churn_by_internet_service.png", "Churn by Internet Service"),
            ("churn_by_payment_method.png", "Churn by Payment Method"),
            ("churn_by_customer_stage.png", "Churn by Customer Stage"),
            ("churn_by_paperless_billing.png", "Churn by Paperless Billing"),
            ("tenure_histogram.png", "Tenure Histogram vs Churn"),
            ("monthly_charges_histogram.png", "Monthly Charges Histogram vs Churn")
        ],
        "🔬 Statistical & Density Distributions": [
            ("correlation_heatmap.png", "Feature Correlation Heatmap"),
            ("boxplots_tenure_monthly_total.png", "Tenure & Charges Box Plots vs Churn"),
            ("kde_tenure.png", "KDE Density Plot: Tenure"),
            ("kde_monthly_charges.png", "KDE Density Plot: Monthly Charges"),
            ("kde_total_charges.png", "KDE Density Plot: Total Charges")
        ]
    }
    
    for category_name, diagrams in diagram_categories.items():
        with st.expander(category_name, expanded=True):
            cols = st.columns(2)
            for idx, (filename, label) in enumerate(diagrams):
                filepath = os.path.join(diagram_dir, filename)
                col = cols[idx % 2]
                with col:
                    if os.path.exists(filepath):
                        st.image(filepath, caption=label, use_container_width=True)
                    else:
                        st.warning(f"Diagram {filename} not found.")

elif app_mode == "🤖 Model Benchmarks":
    st.subheader("🤖 Machine Learning Model Benchmarks & Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Random Forest Classifier</div>
            <div class="metric-val">80.19%</div>
            <div style="color: #9ca3af; margin-top: 6px;">Evaluation Accuracy Score</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - **Estimators**: 200 trees
        - **Max Depth**: 8
        - **Random State**: 42
        - **Key Strengths**: Robust against overfitting, handling non-linear feature interactions.
        """)
        
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Gradient Boosting Classifier</div>
            <div class="metric-val">80.26%</div>
            <div style="color: #9ca3af; margin-top: 6px;">Evaluation Accuracy Score</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - **Estimators**: 200 trees
        - **Learning Rate**: 0.05
        - **Max Depth**: 3
        - **Random State**: 42
        - **Key Strengths**: Sequentially optimizes residual errors, highly accurate probability calibration.
        """)
        
    st.markdown("---")
    st.markdown("### 🏆 Feature Importance Highlights")
    st.markdown("""
    1. **Contract Type (Month-to-Month)**: Highest predictive importance for churn.
    2. **Customer Tenure**: Strong inverse relationship with churn probability.
    3. **Monthly Charges & Total Charges**: Financial commitment directly impacts customer retention.
    4. **Internet Service (Fiber Optic)**: High influence due to price competition.
    5. **Tech Support & Online Security**: Strongest protective retention features.
    """)

