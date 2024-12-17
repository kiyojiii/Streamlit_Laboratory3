import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit app title
st.title("Classification Model Comparison with Sampling Techniques")

# Preload the dataset
@st.cache_data
def load_data():
    path = "C:/Users/user/Desktop/jeah/ITD105/LABORATORY3/csv/survey lung cancer.csv"
    return read_csv(path)

# Load dataset
dataframe = load_data()

# Raw Data Preview in Expander
with st.expander("### Raw Data Preview"):
    st.write(dataframe)

# Preprocessing to handle non-numeric columns
numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
non_numeric_columns = dataframe.select_dtypes(exclude=["number"]).columns.tolist()

if non_numeric_columns:
    st.warning(f"Non-numeric columns detected: {non_numeric_columns}. They will be encoded or excluded.")
    for col in non_numeric_columns:
        if dataframe[col].nunique() < 20:  # Encode categorical columns with fewer unique values
            dataframe[col] = dataframe[col].astype("category").cat.codes
        else:
            dataframe.drop(columns=[col], inplace=True)

# Show preprocessed data
with st.expander("### Preprocessed Preview"):
    st.write(dataframe)

# Default target and feature columns
target_column = st.selectbox("Select the target column (Y):", dataframe.columns, index=dataframe.columns.get_loc("LUNG_CANCER"))
feature_columns = st.multiselect(
    "Select the feature columns (X):",
    [col for col in dataframe.columns if col != target_column],
    default=[col for col in dataframe.columns if col != "LUNG_CANCER"]
)

if not feature_columns:
    st.error("Please select at least one feature column.")
    st.stop()

X = dataframe[feature_columns].values
Y = dataframe[target_column].values

# Sampling technique selection
st.sidebar.header("Sampling Technique")
sampling_technique = st.sidebar.radio("Select Sampling Technique:", ["Train-Test Split", "K-Fold Cross-Validation"])

# K-Fold slider
if sampling_technique == "K-Fold Cross-Validation":
    k = st.sidebar.slider("Number of Folds (K):", 2, 10, 5)

# Train-test split parameters
test_size = st.sidebar.slider("Test Size (fraction):", 0.1, 0.5, 0.2)
random_seed = st.sidebar.slider("Random Seed:", 1, 100, 42)

# Model selection using multi-select
st.sidebar.header("Select Algorithms to Compare")
all_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
    "Gaussian Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(random_state=random_seed),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=random_seed),
    "MLP Classifier": MLPClassifier(max_iter=200, random_state=random_seed),
    "Perceptron": Perceptron(random_state=random_seed),
    "Random Forest": RandomForestClassifier(random_state=random_seed),
    "Support Vector Machine (SVM)": SVC(random_state=random_seed),
}

selected_models = st.sidebar.multiselect(
    "Select Models:", list(all_models.keys()), default=list(all_models.keys())
)

# Remove unselected models
models = {name: model for name, model in all_models.items() if name in selected_models}

# Model evaluation
st.write("### Model Performance Comparison")
results = []

if sampling_technique == "Train-Test Split":
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)
    for model_name, model in models.items():
        model.fit(X_train, Y_train)
        accuracy = model.score(X_test, Y_test) * 100
        results.append({"Model": model_name, "Accuracy": accuracy})

elif sampling_technique == "K-Fold Cross-Validation":
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    for model_name, model in models.items():
        cv_results = cross_val_score(model, X, Y, cv=kf, scoring="accuracy")
        accuracy = np.mean(cv_results) * 100
        results.append({"Model": model_name, "Accuracy": accuracy})

# Display results
if results:
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    st.dataframe(results_df)

    # Bar Chart
    st.write("### Model Accuracy Bar Chart")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))
    plt.barh(results_df["Model"], results_df["Accuracy"], color=colors)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Model")
    plt.title("Model Performance Comparison (Bar Chart)")
    plt.gca().invert_yaxis()
    st.pyplot(plt)

    # Line Chart (Fix for multi-dimensional indexing)
    st.write("### Model Accuracy Line Chart")
    plt.figure(figsize=(10, 6))
    models_array = np.array(results_df["Model"])  # Fix for Pandas multi-dimensional indexing issue
    plt.plot(models_array, results_df["Accuracy"].values, marker='o', linestyle='-', color='blue')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Performance Comparison (Line Chart)")
    plt.tight_layout()
    st.pyplot(plt)

    # Heatmap
    st.write("### Model Performance Heatmap")
    pivot_table = results_df.pivot_table(values="Accuracy", index="Model")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Model Performance Heatmap")
    st.pyplot(plt)

else:
    st.write("No results to display. Please check your settings.")

st.success("Analysis Complete!")

# Conclusion
st.subheader("Conclusion")
if not results_df.empty:
    best_model = results_df.iloc[0]["Model"]
    best_accuracy = results_df.iloc[0]["Accuracy"]
    st.write(
        f"The **{best_model}** model achieved the highest accuracy of **{best_accuracy:.2f}%**. "
        "This suggests that it is the most effective model for this dataset given the selected features and sampling technique."
    )
else:
    st.write("No valid results were produced. Please adjust your settings and try again.")
