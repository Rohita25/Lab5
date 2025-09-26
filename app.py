import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Predictor")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV (semicolon separated)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

st.write("### Dataset Preview", df.head())

# ------------------- Data Exploration ------------------- #
st.subheader("📊 Data Exploration")
col = st.selectbox("Select a column to visualize", df.columns)

fig, ax = plt.subplots()
if df[col].dtype == "object":
    df[col].value_counts().plot(kind="bar", ax=ax, color='skyblue')
    ax.set_ylabel("Count")
else:
    df[col].plot(kind="hist", bins=20, ax=ax, color='skyblue')
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
ax.set_title(f"Distribution of {col}")
st.pyplot(fig)

# ------------------- Preprocess Data ------------------- #
df_model = df.copy()
label_encoders = {}
for c in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[c] = le.fit_transform(df_model[c])
    label_encoders[c] = le  # store encoder for prediction

X = df_model.drop(columns=["G3"], errors="ignore")
y = (df_model["G3"] >= 10).astype(int)  # Pass if G3 >= 10

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ------------------- Prediction Form ------------------- #
st.subheader("🧑‍🎓 Predict Student Outcome")

st.markdown("Fill in the student details below:")
cols = st.columns(2)  # split form into 2 columns for better layout
sample = {}

for i, col_name in enumerate(X.columns):
    container = cols[i % 2]
    dtype = df[col_name].dtype
    
    if dtype == "object":
        sample[col_name] = container.selectbox(col_name, df[col_name].unique())
    elif "int" in str(dtype) or "float" in str(dtype):
        # Use integer input for age/whole numbers, float input for other numeric columns if needed
        if col_name.lower() in ["age", "absences"]:
            sample[col_name] = container.number_input(col_name, int(df[col_name].min()), int(df[col_name].max()), int(df[col_name].mean()))
        else:
            sample[col_name] = container.number_input(col_name, float(df[col_name].min()), float(df[col_name].max()), float(df[col_name].mean()))
            
if st.button("Predict"):
    input_df = pd.DataFrame([sample])
    
    # Encode categorical columns
    for c in input_df.select_dtypes(include='object').columns:
        input_df[c] = label_encoders[c].transform(input_df[c])
    
    prob = model.predict_proba(input_df)[0][1]
    pred = "✅ Pass" if prob >= 0.5 else "❌ Fail"
    
    st.success(f"**Prediction:** {pred}")
    st.info(f"**Probability of passing:** {prob:.2f}")
