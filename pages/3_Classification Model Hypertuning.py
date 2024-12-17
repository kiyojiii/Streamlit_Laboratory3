import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import io
import os

# Streamlit app title
st.title("Classification Model Hyperparameter Tuning with Sampling Techniques")

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

# Target and Feature selection
target_column = st.selectbox("Select target column (Y):", dataframe.columns, index=dataframe.columns.get_loc("LUNG_CANCER"))
feature_columns = st.multiselect(
    "Select feature columns (X):",
    [col for col in dataframe.columns if col != target_column],
    default=[col for col in dataframe.columns if col != target_column]
)

if not feature_columns:
    st.error("Please select at least one feature column.")
    st.stop()

X = dataframe[feature_columns].values
Y = dataframe[target_column].values

# Sampling technique selection
st.sidebar.header("Sampling Technique")
sampling_technique = st.sidebar.radio("Select Sampling Technique:", ["Train-Test Split", "K-Fold Cross-Validation"])

# Train-test split parameters
test_size = st.sidebar.slider("Test Size (fraction):", 0.1, 0.5, 0.2)
random_seed = st.sidebar.slider("Random Seed:", 1, 100, 42)

if sampling_technique == "K-Fold Cross-Validation":
    k = st.sidebar.slider("Number of Folds (K):", 2, 10, 5)

# Models and hyperparameter grids
models = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=random_seed),
        "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    },
    "Gaussian Naive Bayes": {
        "model": GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=random_seed),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "algorithm": ["auto", "ball_tree"]},
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=200, random_state=random_seed),
        "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    },
    "MLP Classifier": {
        "model": MLPClassifier(max_iter=200, random_state=random_seed),
        "params": {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "activation": ["relu", "tanh"], "learning_rate": ["constant"]},
    },
    "Perceptron": {
        "model": Perceptron(random_state=random_seed),
        "params": {"eta0": [0.01, 0.1, 1], "max_iter": [200, 500]},
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=random_seed),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
    },
    "Support Vector Machine (SVM)": {
        "model": SVC(random_state=random_seed),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    },
}

# Hyperparameter Tuning
tuned_results = []

if sampling_technique == "Train-Test Split":
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)
    for model_name, details in models.items():
        st.write(f"Tuning {model_name}...")
        grid_search = GridSearchCV(estimator=details["model"], param_grid=details["params"], cv=3, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        best_accuracy = best_model.score(X_test, Y_test)
        tuned_results.append({"Model": model_name, "Best Parameters": grid_search.best_params_, "Accuracy (%)": round(best_accuracy * 100, 2)})

elif sampling_technique == "K-Fold Cross-Validation":
    for model_name, details in models.items():
        st.write(f"Tuning {model_name} with {k}-Fold Cross-Validation...")
        grid_search = GridSearchCV(estimator=details["model"], param_grid=details["params"], cv=k, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X, Y)
        tuned_results.append({"Model": model_name, "Best Parameters": grid_search.best_params_, "Accuracy (%)": round(np.mean(grid_search.cv_results_['mean_test_score']) * 100, 2)})

# Display Results
tuned_results_df = pd.DataFrame(tuned_results).sort_values(by="Accuracy (%)", ascending=False)
st.write("### Tuned Model Performance Comparison")
st.dataframe(tuned_results_df)

# Visualization - Bar Graph
st.write("### Hypyertuning Visualization")
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(tuned_results_df)))
plt.barh(tuned_results_df["Model"], tuned_results_df["Accuracy (%)"], color=colors)
plt.xlabel("Accuracy (%)")
plt.title("Tuned Model Performance Comparison")
plt.gca().invert_yaxis()
st.pyplot(plt)

# Line Graph - Model Accuracy Trends
st.write("### Accuracy Trends Across Models")
fig, ax = plt.subplots(figsize=(10, 6))

# Convert model names to indices and use for plotting
x_axis = np.arange(len(tuned_results_df))  # Create numerical indices
accuracy = tuned_results_df["Accuracy (%)"].to_numpy()  # Convert Accuracy column to NumPy array

plt.plot(x_axis, accuracy, marker="o", linestyle="-")
plt.xticks(x_axis, tuned_results_df["Model"], rotation=45)  # Replace tick marks with model names
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Trends for Tuned Models")
plt.tight_layout()
st.pyplot(fig)

# Best Model Conclusion
best_model = tuned_results_df.iloc[0]
st.write("### Conclusion")
st.write(
    f"The **best-performing model** is **{best_model['Model']}** with an accuracy of "
    f"**{best_model['Accuracy (%)']}%**. The best hyperparameters are:"
)
st.json(best_model["Best Parameters"])

st.success("Hyperparameter Tuning Complete!")




