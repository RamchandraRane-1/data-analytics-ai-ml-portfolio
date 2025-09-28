import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cohere
import re
import textwrap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 🔐 Cohere setup
cohere_api_key = "C2UZIwWoSszxjxoxcd21YqAaW168jtfylg9KjI2y"
co = cohere.Client(cohere_api_key)

# 📄 Load dataset
df = pd.read_csv("heart_cleaned.csv")

# 🧠 ML Model for prediction
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']
X_encoded = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
feature_names = X_encoded.columns.tolist()

# 🧪 Sample queries with visualization intents
sample_queries = [
    # Bar Chart Oriented
    "Show patients with heart disease",
    "List female patients over 50",
    "Filter patients with cholesterol over 200",
    "Show patients with heart disease and age > 60",
    "List diabetic patients who smoke",

    # Pie Chart Queries
    "Show pie chart of heart disease by sex",
    "Pie chart of chest pain type distribution",

    # Histogram/Distribution
    "Show distribution of cholesterol levels",
    "Histogram of resting blood pressure",

    # Group Summary
    "Show count of patients by ST slope",
    "Group patients by exercise-induced angina and show proportion"
]

# 🧼 Sanitize generated code
def sanitize_pandas_code(code):
    pattern = r"df\s*=\s*df\[(.*?)\]"
    match = re.search(pattern, code)
    if match:
        condition = match.group(1)
        condition_fixed = re.sub(
            r"([\w'\[\]]+\s*[<>=!]=?\s*[\w'\d\"]+)",
            r"(\1)", condition
        )
        condition_fixed = condition_fixed.replace("&", " & ").replace("|", " | ")
        code = re.sub(pattern, f"df = df[{condition_fixed}]", code)
    return code

# 🧠 Generate code using Cohere
def generate_pandas_code(nl_query, columns):
    prompt = f"""
You are a helpful assistant that writes pandas code for analyzing a DataFrame.
The DataFrame has the following columns: {columns}

User query: "{nl_query}"

Always return just one line of pandas code that filters or groups the DataFrame named 'df'.
The line should always be like: df = df[<some condition>] or df = df.groupby(...)

Do NOT include import statements, print(), or any extra comments or example DataFrames.
Just the pandas operation in one line.
"""
    try:
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=200,
            temperature=0.3
        )
        response_text = response.generations[0].text.strip()

        # Extract first valid 'df =' line
        match = re.search(r"df\s*=\s*.+", response_text)
        return match.group(0).strip() if match else "# No valid df line returned"

    except Exception as e:
        return f"# Error generating code: {e}"


# 🧮 Execute query and return chart + explanation
def process_query(nl_query):
    columns = list(df.columns)
    pandas_code = generate_pandas_code(nl_query, columns)

    if pandas_code.startswith("# Error"):
        return None, pandas_code

    pandas_code = sanitize_pandas_code(pandas_code)
    local_vars = {'df': df.copy(), 'result': None}

    try:
        # Check if this is a plotting command (e.g. includes `.plot.`)
        is_plot = ".plot." in pandas_code or ".hist(" in pandas_code or ".pie(" in pandas_code

        if is_plot:
            wrapped_code = textwrap.dedent(f"""
                import matplotlib.pyplot as plt
                {pandas_code}
                plt.tight_layout()
                plt.savefig("plot.png")
                plt.close()
            """)
            exec(wrapped_code, {"pd": pd, "plt": plt}, local_vars)
            return "plot.png", f"🖼️ Generated a plot based on the query.\n\n🧾 Code:\n{pandas_code}"

        else:
            wrapped_code = textwrap.dedent(f"""
                {pandas_code}
                result = df.copy() if isinstance(df, pd.DataFrame) else None
            """)
            exec(wrapped_code, {"pd": pd}, local_vars)

            result_df = local_vars['result']
            explanation = ""

            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                num_rows = len(result_df)
                explanation += f"✅ Query filtered the dataset to {num_rows} records.\n"

                if 'HeartDisease' in result_df.columns:
                    heart_disease_count = result_df['HeartDisease'].sum()
                    explanation += f"🫀 {heart_disease_count} patients have heart disease in the filtered data.\n"

                numeric_cols = result_df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    top_col = numeric_cols[0]
                    top_vals = result_df[top_col].value_counts().head(3).to_dict()
                    top_vals_str = ", ".join([f"{k}: {v}" for k, v in top_vals.items()])
                    explanation += f"📊 Top 3 values in `{top_col}`: {top_vals_str}\n"

                    plt.figure(figsize=(10, 6))
                    ax = result_df[top_col].value_counts().head(5).plot(kind='bar', color='skyblue', edgecolor='black')
                    plt.title(f"Top 5 Values in '{top_col}'", fontsize=14)
                    plt.xlabel(top_col)
                    plt.ylabel("Count")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    for p in ax.patches:
                        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height() + 1),
                                    ha='center')
                    plt.tight_layout()
                    plt.legend([top_col])
                    plt.savefig("plot.png")
                    plt.close()
                    return "plot.png", f"{explanation}\n\n🧾 Code:\n{pandas_code}"
                else:
                    return None, f"{explanation}\n⚠️ No numeric columns to plot.\n\n🧾 Code:\n{pandas_code}"
            else:
                return None, f"Query returned no data.\n\n🧾 Code:\n{pandas_code}"

    except Exception as e:
        return None, f"❌ Error executing code:\n{pandas_code}\n\nException: {e}"


# 🔮 Predict heart disease
def predict_heart_disease(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability

# 🌐 Streamlit UI
st.set_page_config(page_title="Heart Failure Dashboard", layout="wide")
st.title("Heart Failure NLQ")

# 🔍 Sidebar info
with st.sidebar:
    st.header("📋 Dataset Info")
    st.dataframe(df.dtypes.astype(str))
    st.markdown("---")
    st.header("💡 Sample Queries")
    selected = st.selectbox("Try a sample:", sample_queries)
    if st.button("Use This"):
        st.session_state["query"] = selected

# 🔄 Dual-panel interface
col1, col2 = st.columns(2)

# Left: Structured NLQ
with col1:
    st.subheader("📊 Structured NLQ")
    query = st.text_input("Ask about your data:", value=st.session_state.get("query", ""), key="structured")
    if query:
        plot_path, message = process_query(query)
        if plot_path:
            st.image(plot_path, caption="Generated Plot", use_column_width=True)
        st.markdown(f"```\n{message}\n```")

# Right: Conversational Q&A
with col2:
    st.subheader("💬 Conversational Q&A")
    chat_q = st.text_input("Ask a general question:", key="chat")
    if chat_q:
        response = co.generate(
            model="command-r-plus",
            prompt=f"You are a helpful data analyst working with a dataset with these columns: {list(df.columns)}.\nAnswer this user question clearly:\n\n{chat_q}",
            max_tokens=300,
            temperature=0.5
        )
        st.markdown(f"**Answer:** {response.generations[0].text.strip()}")

# 🔮 Prediction Section
st.markdown("---")
st.header("🔍 Predict Heart Disease Risk")

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
    sex = st.selectbox("Sex", ["F", "M"])
    cp = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])
    predict = st.form_submit_button("Predict Risk")

    if predict:
        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_" + sex: 1,
            "ChestPainType_" + cp: 1,
            "RestingECG_" + ecg: 1,
            "ExerciseAngina_" + angina: 1,
            "ST_Slope_" + slope: 1
        }
        prediction, prob = predict_heart_disease(input_data)
        label = "❗ Likely to have heart disease" if prediction == 1 else "✅ Unlikely to have heart disease"
        st.success(f"**Prediction:** {label} (Probability: {prob:.2f})")
