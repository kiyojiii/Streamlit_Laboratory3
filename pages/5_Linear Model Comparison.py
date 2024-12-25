import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Streamlit app title
st.title("Regression Model Comparison")

# Preload weather.csv if no file is uploaded
default_csv_path = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\csv\weather.csv"

if os.path.exists(default_csv_path):
    dataframe = read_csv(default_csv_path)
    st.success("Dataset successfully loaded from predefined path.")
else:
    st.error(f"Default dataset not found at {default_csv_path}")
    st.stop()

# Display raw dataset
st.write("### Dataset Preview")
with st.expander("Show Raw Dataset", expanded=False):
    st.dataframe(dataframe)

# Preprocessing non-numeric columns
non_numeric_columns = dataframe.select_dtypes(exclude=["number"]).columns.tolist()
for col in non_numeric_columns:
    if dataframe[col].nunique() < 20:
        dataframe[col] = dataframe[col].astype("category").cat.codes
    else:
        dataframe.drop(columns=[col], inplace=True)

# Display processed dataset
st.write("### Processed Dataset Preview")
with st.expander("Show Processed Dataset Preview", expanded=False):
    st.dataframe(dataframe)

# Target and feature selection
available_targets = ["Temperature_c"]
selected_target_columns = st.multiselect("Select target columns (Y):", available_targets, default=available_targets)
feature_columns = st.multiselect("Select feature columns (X):",
                                 [col for col in dataframe.columns if col not in selected_target_columns],
                                 default=[col for col in dataframe.columns if col not in selected_target_columns])

if selected_target_columns and feature_columns:
    # Sampling technique
    sampling_technique = st.sidebar.radio("Choose Sampling Technique:", ["Train-Test Split", "K-Fold Cross-Validation"])
    max_sample_size = st.sidebar.number_input("Max Rows for Evaluation", min_value=100, max_value=len(dataframe), value=min(1000, len(dataframe)))
    test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
    random_seed = st.sidebar.slider("Random Seed", 1, 100, 42)

    sampled_df = dataframe.sample(n=max_sample_size, random_state=random_seed)
    X = sampled_df[feature_columns]
    Y = sampled_df[selected_target_columns]

    if sampling_technique == "Train-Test Split":
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)
    else:
        n_folds = st.sidebar.slider("Number of Folds for K-Fold", 2, 10, 5)

    # Model selection
    available_models = {
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_seed),
        "Elastic Net": ElasticNet(random_state=random_seed),
        "AdaBoost Regressor": AdaBoostRegressor(random_state=random_seed),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
        "Lasso Regression": Lasso(random_state=random_seed),
        "Ridge Regression": Ridge(random_state=random_seed),
        "Linear Regression": LinearRegression(),
        "MLP Regressor": MLPRegressor(random_state=random_seed, max_iter=500),
        "Random Forest Regressor": RandomForestRegressor(random_state=random_seed),
        "Support Vector Regressor (SVR)": SVR(),
    }

    selected_models = st.sidebar.multiselect("Select Models to Evaluate", list(available_models.keys()), default=list(available_models.keys()))
    models = {name: available_models[name] for name in selected_models}

    # Model evaluation
    results = []
    with st.spinner("Evaluating models..."):
        for model_name, model in models.items():
            try:
                if sampling_technique == "Train-Test Split":
                    Y_train_array = Y_train.values.ravel() if Y_train.shape[1] == 1 else Y_train.values
                    Y_test_array = Y_test.values.ravel() if Y_test.shape[1] == 1 else Y_test.values
                    model.fit(X_train, Y_train_array)
                    predictions = model.predict(X_test).ravel()
                    mae = np.abs(predictions - Y_test_array).mean()
                else:  # K-Fold Cross Validation
                    mae = -cross_val_score(model, X, Y.values.ravel(), scoring="neg_mean_absolute_error",
                                           cv=KFold(n_splits=n_folds, shuffle=True, random_state=random_seed),
                                           n_jobs=-1).mean()
                results.append({"Model": model_name, "Mean Absolute Error (MAE)": mae})
            except Exception as e:
                st.error(f"Error evaluating {model_name}: {e}")

    # Results table
    results_df = pd.DataFrame(results).sort_values(by="Mean Absolute Error (MAE)", ascending=True)
    st.write("### Regression Model Performance Comparison")
    st.dataframe(results_df)

    # Bar graph
    st.write("### Model Performance Bar Chart")
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["Model"], results_df["Mean Absolute Error (MAE)"].to_numpy(), color=plt.cm.tab10(np.linspace(0, 1, len(results_df))))
    plt.xlabel("Mean Absolute Error (MAE)")
    plt.title("Model Performance Comparison")
    plt.gca().invert_yaxis()
    st.pyplot(plt)

    # Line graph
    st.write("### Model Performance Line Chart")
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["Model"].tolist(), results_df["Mean Absolute Error (MAE)"].to_numpy(), marker='o', linestyle='-', color='blue')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Model Performance Comparison")
    st.pyplot(plt)

    # Heatmap
    st.write("### Model Performance Heatmap")
    heatmap_data = results_df.pivot_table(values="Mean Absolute Error (MAE)", index="Model")
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", cbar=True, fmt=".3f", linewidths=0.5)
    plt.title("Model Performance Heatmap")
    st.pyplot(plt)

    st.success("Analysis Complete!")

    # Conclusion
    st.subheader("Conclusion")
    if not results_df.empty:
        best_model = results_df.iloc[0]["Model"]
        best_mae = results_df.iloc[0]["Mean Absolute Error (MAE)"]
        st.write(
            f"The **{best_model}** model achieved the lowest Mean Absolute Error of **{best_mae:.3f}**. "
            "This suggests that it is the most effective model for this dataset given the selected features and sampling technique."
        )
    else:
        st.write("No valid results were produced. Please adjust your settings and try again.")
