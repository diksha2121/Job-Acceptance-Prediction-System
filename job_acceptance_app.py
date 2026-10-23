import streamlit as st
import pandas as pd
import numpy as np

# --- CRITICAL FIX FOR STREAMLIT PLOTS ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page configuration
st.set_page_config(
    page_title="HR Job Acceptance & ML Dashboard",
    page_icon="👔",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-text { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #2563EB; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("HR_Job_Placement_Dataset.csv")
    df_clean = df.copy()

    # Fill missing values
    df_clean['relevant_experience'] = df.loc[df_clean.index, 'relevant_experience']
    str_cols = ['relocation_willingness', 'layoff_history', 'career_switch_willingness', 'relevant_experience', 'job_role_match']
    for col in str_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    num_cols = ['ssc_percentage', 'hsc_percentage', 'notice_period_days', 'employment_gap_months']
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    cat_cols = df_clean.select_dtypes(include='object').columns.to_list()
    for col in cat_cols:
        if col != 'status':
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip().str.title()

    # Standardize target status column terminology
    if 'status' in df_clean.columns:
        df_clean['status'] = df_clean['status'].astype(str).str.strip().str.title()
        status_map = {
            'Placed': 'Job Accepted',
            'Not Placed': 'Job Not Accepted',
            'Job Accepted': 'Job Accepted',
            'Job Not Accepted': 'Job Not Accepted'
        }
        df_clean['status'] = df_clean['status'].map(status_map).fillna('Job Not Accepted')

    # Logical consistency checks
    df_clean['age_years'] = pd.to_numeric(df_clean['age_years'], errors='coerce')
    df_clean['years_of_experience'] = pd.to_numeric(df_clean['years_of_experience'], errors='coerce')

    percentage_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'skills_match_percentage']
    for col in percentage_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean = df_clean[(df_clean[col].isna()) | ((df_clean[col] >= 0) & (df_clean[col] <= 100))]

    df_clean = df_clean[(df_clean['age_years'].isna()) | (df_clean['age_years'] >= 18)]
    df_clean = df_clean[
        (df_clean['years_of_experience'].isna()) |
        (df_clean['age_years'].isna()) |
        (df_clean['years_of_experience'] <= (df_clean['age_years'] - 18))
    ]

    non_negative_cols = ['previous_ctc_lpa', 'expected_ctc_lpa', 'notice_period_days', 'employment_gap_months', 'certifications_count']
    for col in non_negative_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean = df_clean[(df_clean[col].isna()) | (df_clean[col] >= 0)]

    if 'years_of_experience' in df_clean.columns and 'previous_ctc_lpa' in df_clean.columns:
        df_clean.loc[df_clean['years_of_experience'] == 0, 'previous_ctc_lpa'] = 0

    # Feature Engineering
    def experience_category(years):
        if years == 0: 
            return 'Fresher'
        elif years <= 3: 
            return 'Junior'
        else: 
            return 'Senior'

    df_clean['experience_category'] = df_clean['years_of_experience'].apply(experience_category)
    df_clean['experience_category'] = pd.Categorical(df_clean['experience_category'], categories=['Fresher', 'Junior', 'Senior'], ordered=True)

    df_clean['academic_percentage'] = (df_clean['ssc_percentage'] + df_clean['hsc_percentage'] + df_clean['degree_percentage']) / 3
    def academic_performance(pct):
        if pct < 60: 
            return 'Low'
        elif pct <= 75: 
            return 'Average'
        else: 
            return 'High'

    df_clean['academic_performance'] = df_clean['academic_percentage'].apply(academic_performance)
    df_clean['academic_performance'] = pd.Categorical(df_clean['academic_performance'], categories=['Low', 'Average', 'High', 'Unknown'], ordered=True)

    def skills_match_level(pct):
        if pct < 50: 
            return 'Low'
        elif pct <= 75: 
            return 'Medium'
        else: 
            return 'High'

    df_clean['skills_match_level'] = df_clean['skills_match_percentage'].apply(skills_match_level)
    df_clean['skills_match_level'] = pd.Categorical(df_clean['skills_match_level'], categories=['Low', 'Medium', 'High', 'Unknown'], ordered=True)

    df_clean['avg_interview_score'] = (df_clean['technical_score'] + df_clean['aptitude_score'] + df_clean['communication_score']) / 3
    def interview_performance_category(score):
        if score < 50: 
            return 'Needs Improvement'
        elif score <= 75: 
            return 'Good'
        else: 
            return 'Excellent'

    df_clean['interview_performance_category'] = df_clean['avg_interview_score'].apply(interview_performance_category)
    df_clean['interview_performance_category'] = pd.Categorical(
        df_clean['interview_performance_category'], categories=['Needs Improvement', 'Good', 'Excellent', 'Unknown'], ordered=True)

    def placement_probability_score(row):
        return ((row['skills_match_percentage'] * 0.4) + (row['avg_interview_score'] * 0.4) + (row['academic_percentage'] * 0.2)) / 100

    df_clean['placement_probability_score'] = df_clean.apply(placement_probability_score, axis=1)

    # Feature Selection for Modeling
    cols_to_drop = [
        'ssc_percentage', 'hsc_percentage', 'degree_percentage',
        'technical_score', 'aptitude_score', 'communication_score',
        'experience_category', 'skills_match_level',
        'academic_performance', 'interview_performance_category'
    ]
    df_final = df_clean.drop(columns=cols_to_drop, errors='ignore').copy()

    # Train-test split & encoding with updated target mapping
    X = df_final.drop(columns=['status'])
    y = df_final['status'].map({'Job Not Accepted': 0, 'Job Accepted': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ordinal_cols = ['company_tier', 'competition_level']
    nominal_cols = [
        'gender', 'degree_specialization', 'career_switch_willingness',
        'relevant_experience', 'job_role_match', 'bond_requirement',
        'layoff_history', 'relocation_willingness', 'internship_experience'
    ]

    tier_mapping = ['Tier 3', 'Tier 2', 'Tier 1']
    competition_mapping = ['Low', 'Medium', 'High']
    ordinal_encoder = OrdinalEncoder(categories=[tier_mapping, competition_mapping])

    X_train[ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])
    X_test[ordinal_cols] = ordinal_encoder.transform(X_test[ordinal_cols])

    oh_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    train_encoded_oh = oh_encoder.fit_transform(X_train[nominal_cols])
    test_encoded_oh = oh_encoder.transform(X_test[nominal_cols])

    encoded_feature_names = oh_encoder.get_feature_names_out(nominal_cols)
    X_train_oh = pd.DataFrame(train_encoded_oh, columns=encoded_feature_names, index=X_train.index)
    X_test_oh = pd.DataFrame(test_encoded_oh, columns=encoded_feature_names, index=X_test.index)

    X_train = X_train.drop(columns=nominal_cols)
    X_test = X_test.drop(columns=nominal_cols)

    X_train = pd.concat([X_train, X_train_oh], axis=1)
    X_test = pd.concat([X_test, X_test_oh], axis=1)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 1. Base Models (Default parameters)
    model1_base = DecisionTreeClassifier(random_state=42)
    model1_base.fit(X_train_smote, y_train_smote)
    y_base_pred1 = model1_base.predict(X_test)

    model2_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    model2_base.fit(X_train_smote, y_train_smote)
    y_base_pred2 = model2_base.predict(X_test)

    xgbmodel_base = XGBClassifier(random_state=42)
    xgbmodel_base.fit(X_train_smote, y_train_smote)
    y_pred_xgb_base = xgbmodel_base.predict(X_test)

    # 2. Tuned / Pruned Models
    model1_pruned = DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_split=50, min_samples_leaf=20, random_state=42)
    model1_pruned.fit(X_train_smote, y_train_smote)
    y_prune_pred1 = model1_pruned.predict(X_test)

    model2_pruned = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_split=30, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model2_pruned.fit(X_train_smote, y_train_smote)
    y_prune_pred2 = model2_pruned.predict(X_test)

    xgbmodel_tuned = XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    xgbmodel_tuned.fit(X_train_smote, y_train_smote)
    y_pred_xgb_tuned = xgbmodel_tuned.predict(X_test)

    return (df_clean, df_final, X_test, y_test, 
            model1_base, model1_pruned, 
            model2_base, model2_pruned, 
            xgbmodel_base, xgbmodel_tuned, 
            y_base_pred1, y_prune_pred1, 
            y_base_pred2, y_prune_pred2, 
            y_pred_xgb_base, y_pred_xgb_tuned, 
            X_train_smote, y_train_smote)

with st.spinner("🚀 Loading dataset and training Base & Tuned ML models..."):
    (df_clean, df_final, X_test, y_test, 
     model1_base, model1_pruned, 
     model2_base, model2_pruned, 
     xgbmodel_base, xgbmodel_tuned, 
     y_base_pred1, y_prune_pred1, 
     y_base_pred2, y_prune_pred2, 
     y_pred_xgb_base, y_pred_xgb_tuned, 
     X_train_smote, y_train_smote) = load_and_process_data()

# Main Application Layout
st.markdown('<p class="main-header">HR Job Acceptance Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">End-to-end recruitment analytics, predictive modeling performance, and business impact metrics.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Tab 1: Exploratory Data Analysis (EDA)", "🤖 Tab 2: ML Model Performance", "📈 Tab 3: Business & ML KPIs"])

# ================= TAB 1 =================
with tab1:
    st.header("Exploratory Data Analysis")

    # 1. Interview Score Vs Job Acceptance
    st.subheader("1. Interview Score vs. Job Acceptance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = sns.barplot(data=df_clean, x='status', y='avg_interview_score', estimator='mean', errorbar=None, palette='Set2', ax=ax[0])
    for containers in  ax1.containers:
        ax1.bar_label(containers, fmt='%.2f')
    ax[0].set_title('Interview Score vs Job Acceptance (Mean)')
    sns.boxplot(data=df_clean, x='status', y='avg_interview_score', palette='Set2', ax=ax[1])
    ax[1].set_title('Interview Score Spread by Status')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 2. Skills match percentage impact on placement
    st.subheader("2. Skills Match Percentage Impact on Job Acceptance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df_clean, x='skills_match_percentage', hue='status', bins=10, kde=True, palette='viridis', ax=ax[0])
    ax[0].set_title('Skills Match Percentage Distribution')
    ax2 = sns.barplot(data=df_clean, x='status', y='skills_match_percentage', estimator='mean', errorbar=None, palette='viridis', ax=ax[1])
    for containers in ax2.containers:
        ax2.bar_label(containers, fmt='%.2f')
    ax[1].set_title('Average Skills Match Percentage vs Status')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 3. Company tier vs acceptance rate
    st.subheader("3. Company Tier vs. Job Acceptance Status & Rate")
    tier_order = ['Tier 1', 'Tier 2', 'Tier 3']
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax3 = sns.countplot(data=df_clean, x='company_tier', hue='status', order=tier_order, palette='Set2', ax=ax[0])
    for containers in ax3.containers:
        ax3.bar_label(containers, padding=3, fontsize=10, weight="bold")
    ax[0].set_title('Company Tier vs Job Acceptance Status (Counts)')

    ct = pd.crosstab(df_clean['company_tier'], df_clean['status'], normalize='index') * 100
    ct = ct.reindex(tier_order)
    ax4 = ct.plot(kind='bar', stacked=True, ax=ax[1], colormap='viridis')
    for container in ax4.containers:
        labels = [f'{val:.1f}%' for val in container.datavalues]
        ax4.bar_label(container, labels=labels, label_type='center', color='white', weight='bold', fontsize=10)
    ax[1].set_title('Acceptance Rate Proportion by Company Tier (%)')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 4. Experience vs placement probability
    st.subheader("4. Experience vs. Job Acceptance Probability & Success")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df_clean, x='years_of_experience', y='placement_probability_score', palette='Set2', ax=ax[0])
    ax[0].set_title('Job Acceptance Probability Spread by Experience Years')

    sns.boxplot(data=df_clean, x='experience_category', y='placement_probability_score', palette='Set2', ax=ax[1])
    ax[1].set_title('Job Acceptance Probability Distribution by Experience Tier')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 5. Competition level vs job acceptance
    st.subheader("5. Competition Level vs. Job Acceptance")
    level_order = ['Low', 'Medium', 'High']
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax5 = sns.countplot(data=df_clean, x='competition_level', hue='status', order=level_order, palette='Set2', ax=ax[0])
    for container in ax5.containers:
        ax5.bar_label(containers, padding=3, fontsize=10, weight="bold")
    ax[0].set_title('Competition Level vs Job Acceptance')

    ct1 = pd.crosstab(df_clean['competition_level'], df_clean['status'], normalize='index') * 100
    ct1 = ct1.reindex(level_order)
    ax6 = ct1.plot(kind='bar', stacked=True, ax=ax[1], colormap='viridis')
    for container in ax6.containers:
        ax[1].bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, weight = 'bold', color='white')
    ax[1].set_title('Acceptance Rate Proportion by Competition Level (%)')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 6. Academic Score vs Placement Outcome
    st.subheader("6. Academic Score Distribution by Job Acceptance Status")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df_final, x='status', y='academic_percentage', palette='Set2', ax=ax[0])
    ax[0].set_title('Academic Score Distribution by Status')
    sns.histplot(data=df_final, x='academic_percentage', hue='status', bins=25, kde=True, palette='Set2', multiple='stack', ax=ax[1])
    ax[1].set_title('Academic Score Distribution Overlay')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 7. Certification Impact on Job Acceptance
    st.subheader("7. Certification Impact on Job Acceptance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax_cert1 = sns.countplot(data=df_final, x='certifications_count', hue='status', palette='Set2', ax=ax[0])
    for container in ax_cert1.containers:
        ax_cert1.bar_label(container, padding=3, fontsize=9)
    ax[0].set_title('Candidate Count by Certifications & Status')

    cert_cross = pd.crosstab(df_final['certifications_count'], df_final['status'], normalize='index') * 100
    cert_cross.plot(kind='bar', stacked=True, ax=ax[1], colormap='viridis')
    ax[1].set_title('Job Acceptance Rate Proportion by Certifications Count')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # 8. Correlation Analysis among numerical features
    st.subheader("8. Correlation Analysis Among Numerical Features")
    num_col_list = df_clean.select_dtypes(include='number').columns.to_list()
    corr_matrix = df_clean[num_col_list].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(data=corr_matrix, annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1, linewidth=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, pad=15, weight='bold')
    st.pyplot(fig)
    plt.close(fig)

# ================= TAB 2 =================
with tab2:
    st.header("Machine Learning Model Evaluation & Comparison")

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        model_choice = st.selectbox(
            "Select Model for Detailed Inspection", 
            [
                "Base Decision Tree", "Pruned Decision Tree", 
                "Base Random Forest", "Tuned Random Forest", 
                "Base XGBoost", "Tuned XGBoost"
            ]
        )
    with col_ctrl2:
        st.write("")
        run_button = st.button("🚀 Run Inspection", type="primary")

    if run_button or True:
        if model_choice == "Base Decision Tree":
            selected_model = model1_base
            y_p = y_base_pred1
        elif model_choice == "Pruned Decision Tree":
            selected_model = model1_pruned
            y_p = y_prune_pred1
        elif model_choice == "Base Random Forest":
            selected_model = model2_base
            y_p = y_base_pred2
        elif model_choice == "Tuned Random Forest":
            selected_model = model2_pruned
            y_p = y_prune_pred2
        elif model_choice == "Base XGBoost":
            selected_model = xgbmodel_base
            y_p = y_pred_xgb_base
        else:
            selected_model = xgbmodel_tuned
            y_p = y_pred_xgb_tuned

        acc = accuracy_score(y_test, y_p)
        cm = confusion_matrix(y_test, y_p)

        mcol1, mcol2 = st.columns([1, 2])
        with mcol1:
            st.markdown(f"### Model Accuracy")
            st.metric(label="Test Accuracy", value=f"{acc*100:.2f}%")
            
            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

        with mcol2:
            st.markdown("### Classification Report")
            report_dict = classification_report(y_test, y_p, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

        st.markdown("---")
        st.subheader(f"Feature Importance ({model_choice})")
        feat_imp = pd.DataFrame({"Feature": X_train_smote.columns, "Importance": selected_model.feature_importances_})
        feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="crest", ax=ax)
        ax.set_title("Top 10 Feature Importances")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("📊 Base vs. Tuned Models: Training vs. Testing Accuracy Comparison")
    st.markdown("Compare training accuracy against testing accuracy across all Base and Tuned models to demonstrate overfitting reduction.")

    # Calculate Train and Test Accuracies for all 6 models
    dt_base_train = model1_base.score(X_train_smote, y_train_smote) * 100
    dt_base_test = accuracy_score(y_test, y_base_pred1) * 100

    dt_prune_train = model1_pruned.score(X_train_smote, y_train_smote) * 100
    dt_prune_test = accuracy_score(y_test, y_prune_pred1) * 100

    rf_base_train = model2_base.score(X_train_smote, y_train_smote) * 100
    rf_base_test = accuracy_score(y_test, y_base_pred2) * 100

    rf_tuned_train = model2_pruned.score(X_train_smote, y_train_smote) * 100
    rf_tuned_test = accuracy_score(y_test, y_prune_pred2) * 100

    xgb_base_train = xgbmodel_base.score(X_train_smote, y_train_smote) * 100
    xgb_base_test = accuracy_score(y_test, y_pred_xgb_base) * 100

    xgb_tuned_train = xgbmodel_tuned.score(X_train_smote, y_train_smote) * 100
    xgb_tuned_test = accuracy_score(y_test, y_pred_xgb_tuned) * 100

    # Summary Table for Presentation
    summary_table = pd.DataFrame({
        'Model Variant': [
            'Base Decision Tree', 'Pruned Decision Tree', 
            'Base Random Forest', 'Tuned Random Forest', 
            'Base XGBoost', 'Tuned XGBoost'
        ],
        'Training Accuracy (%)': [
            f"{dt_base_train:.2f}%", f"{dt_prune_train:.2f}%", 
            f"{rf_base_train:.2f}%", f"{rf_tuned_train:.2f}%", 
            f"{xgb_base_train:.2f}%", f"{xgb_tuned_train:.2f}%"
        ],
        'Testing Accuracy (%)': [
            f"{dt_base_test:.2f}%", f"{dt_prune_test:.2f}%", 
            f"{rf_base_test:.2f}%", f"{rf_tuned_test:.2f}%", 
            f"{xgb_base_test:.2f}%", f"{xgb_tuned_test:.2f}%"
        ],
        'Overfitting Gap (%)': [
            f"{(dt_base_train - dt_base_test):.2f}%", f"{(dt_prune_train - dt_prune_test):.2f}%", 
            f"{(rf_base_train - rf_base_test):.2f}%", f"{(rf_tuned_train - rf_tuned_test):.2f}%", 
            f"{(xgb_base_train - xgb_base_test):.2f}%", f"{(xgb_tuned_train - xgb_tuned_test):.2f}%"
        ]
    })
    st.dataframe(summary_table.set_index('Model Variant'), use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.header("Business KPIs & Machine Learning KPIs")

    # Calculations matching user KPI logic with updated status labels
    total_candidates = len(df_clean)
    accepted_candidates = len(df_clean[df_clean['status'] == 'Job Accepted'])
    overall_acceptance_rate = (accepted_candidates / total_candidates) * 100

    avg_skills_accepted = df_clean[df_clean['status'] == 'Job Accepted']['skills_match_percentage'].mean()
    avg_skills_not_accepted = df_clean[df_clean['status'] == 'Job Not Accepted']['skills_match_percentage'].mean()

    avg_interview_accepted = df_clean[df_clean['status'] == 'Job Accepted']['avg_interview_score'].mean()
    avg_interview_not_accepted = df_clean[df_clean['status'] == 'Job Not Accepted']['avg_interview_score'].mean()

    dt_accuracy = accuracy_score(y_test, y_prune_pred1) * 100
    rf_accuracy = accuracy_score(y_test, y_prune_pred2) * 100
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb_tuned) * 100

    # KPI Top Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Candidates", f"{total_candidates:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Acceptance Rate", f"{overall_acceptance_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Interview (Accepted)", f"{avg_interview_accepted:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Model (XGB)", f"{xgb_accuracy:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_biz, col_ml = st.columns(2)
    with col_biz:
        st.subheader("📊 Business & Dataset Performance KPIs")
        st.write(f"- **Overall Job Acceptance Rate:** {overall_acceptance_rate:.2f}%")
        st.write(f"- **Avg Skills Match (Accepted):** {avg_skills_accepted:.2f}%")
        st.write(f"- **Avg Skills Match (Not Accepted):** {avg_skills_not_accepted:.2f}%")
        st.write(f"- **Avg Interview Score (Accepted):** {avg_interview_accepted:.2f}")
        st.write(f"- **Avg Interview Score (Not Accepted):** {avg_interview_not_accepted:.2f}")

    with col_ml:
        st.subheader("Predictive Model Accuracy KPIs")
        st.write(f"- **Decision Tree (Tuned):** {dt_accuracy:.2f}%")
        st.write(f"- **Random Forest (Tuned):** {rf_accuracy:.2f}%")
        st.write(f"- **XGBoost (Tuned):** {xgb_accuracy:.2f}%")

    st.markdown("---")
    st.subheader("Job Acceptance Probability Spread Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df_clean, x='placement_probability_score', hue='status', kde=True, bins=15, palette='coolwarm', ax=ax)
    ax.set_title("Composite Job Acceptance Probability Spread by Status")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Key Factors Influencing Job Acceptance Decisions")

    # Extract and sort feature importances from Tuned XGBoost
    xgb_importances = pd.DataFrame({
        'Feature': X_train_smote.columns,
        'Importance': xgbmodel_tuned.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    readable_names = {
        'relevant_experience_Relevant': 'Has Relevant Experience',
        'internship_experience_Yes': 'Has Internship Experience',
        'job_role_match_Not Matched': 'Job Role Matched',
        'relocation_willingness_Willing': 'Willing to Relocate',
        'bond_requirement_Required': 'Bond Requirement Enforced',
        'career_switch_willingness_Willing': 'Willing to Switch Career',
        'competition_level': 'Competition Level',
        'layoff_history_Yes': 'Has Layoff History',
        'avg_interview_score': 'Avg Interview Score',
        'previous_ctc_lpa': 'Previous CTC (LPA)',
        'years_of_experience': 'Years of Experience',
        'placement_probability_score': 'Placement Probability Score',
        'skills_match_percentage': 'Skills Match %',
        'expected_ctc_lpa': 'Expected CTC (LPA)',
        'gender_Male': 'Gender (Male)',
        'employment_gap_months': 'Employment Gap (Months)',
        'notice_period_days': 'Notice Period (Days)',
        'academic_percentage': 'Academic Score %',
        'company_tier': 'Company Tier',
        'degree_specialization_Mechanical': 'Degree: Mechanical',
        'certifications_count': 'Certifications Count',
        'degree_specialization_Others': 'Degree: Others',
        'age_years': 'Age (Years)',
        'degree_specialization_Information Technology': 'Degree: IT',
        'degree_specialization_Electronics': 'Degree: Electronics'
    }

    xgb_importances['Feature_Readable'] = xgb_importances['Feature'].map(readable_names).fillna(xgb_importances['Feature'])

    # Plot top factors
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=xgb_importances.head(15),
        x='Importance',
        y='Feature_Readable',
        palette='viridis',
        ax=ax
    )
    ax.set_title('Top 15 Key Factors Influencing Job Acceptance (XGBoost Feature Importance)', fontsize=13, weight='bold', pad=15)
    ax.set_xlabel('Relative Importance Weight', fontsize=11)
    ax.set_ylabel('Influencing Factor', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Top Factors Summary Table")
    st.dataframe(xgb_importances[['Feature_Readable', 'Importance']].head(10).reset_index(drop=True).style.format({"Importance": "{:.4f}"}))