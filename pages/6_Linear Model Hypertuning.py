import streamlit as st
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

# Streamlit app title
st.title("Regression Model Hypertuning")

# Preload the dataset from a predefined path
default_csv_path = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\csv\weather.csv"

if os.path.exists(default_csv_path):
    dataframe = read_csv(default_csv_path)
    st.success("Dataset successfully loaded from predefined path.")
else:
    st.error(f"Default dataset not found at {default_csv_path}")
    st.stop()


if default_csv_path:
    # Load the dataset
    dataframe = read_csv(default_csv_path)
    st.write("Dataset Preview:")
    st.write(dataframe.head())


    # Sampling technique selection
    sampling_technique = st.sidebar.radio("Choose Sampling Technique", ["Train-Test Split", "K-Fold Cross-Validation"])
    
    # Data Sampling for Faster Execution
    sample_fraction = st.sidebar.slider("Sample Fraction", 0.1, 1.0, 0.5)
    dataframe = dataframe.sample(frac=sample_fraction, random_state=42)

    # Preprocessing to handle non-numeric columns
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_columns = dataframe.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_columns:
        st.warning(f"Non-numeric columns detected: {non_numeric_columns}. Text columns will be processed.")
        for col in non_numeric_columns:
            if dataframe[col].nunique() < 20:  # Encode categorical columns with fewer unique values
                dataframe[col] = dataframe[col].astype("category").cat.codes
            elif col == "description":  # Preprocess the description column
                st.info(f"Processing text column: {col}")
                vectorizer = TfidfVectorizer(max_features=50)  # Limit to top 50 features
                description_features = vectorizer.fit_transform(dataframe[col].astype(str)).toarray()
                description_feature_names = [f"{col}_{i}" for i in range(description_features.shape[1])]
                description_df = pd.DataFrame(description_features, columns=description_feature_names)
                dataframe = pd.concat([dataframe, description_df], axis=1)
                dataframe.drop(columns=[col], inplace=True)  # Drop the original text column
            else:
                dataframe.drop(columns=[col], inplace=True)  # Drop unsuitable columns

    # Display the preprocessed data
    st.write("### Preprocessed Data")
    st.write(dataframe.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", numeric_columns, index=numeric_columns.index("Temperature_c"))
    feature_columns = st.multiselect("Select Feature Columns", [col for col in numeric_columns if col != target_column], default=[col for col in numeric_columns if col != target_column])

    if target_column in dataframe.columns:
        X = dataframe[feature_columns]
        Y = dataframe[[target_column]]  # Double brackets to keep it as a DataFrame

        # Train-test split
        test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key='test_size')
        random_seed = st.sidebar.slider("Random Seed", 1, 100, 42, key='random_seed')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        # Select models to tune
        all_models = [
            "Decision Tree Regressor", "Elastic Net", "AdaBoost Regressor",
            "K-Nearest Neighbors Regressor", "Lasso Regression", "Ridge Regression",
            "Linear Regression", "MLP Regressor", "Random Forest Regressor",
            "Support Vector Regressor (SVR)"
        ]
        selected_models = st.multiselect("Select Models for Hyperparameter Tuning", all_models, default=all_models)

        # Define models and hyperparameter grids
        models = {
            "Decision Tree Regressor": {
                "model": DecisionTreeRegressor(random_state=random_seed),
                "params": {
                    "max_depth": [3, 5],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1]
                }
            },
            "Elastic Net": {
                "model": ElasticNet(random_state=random_seed),
                "params": {
                    "alpha": [0.1, 1.0],
                    "l1_ratio": [0.2]
                }
            },
            "AdaBoost Regressor": {
                "model": AdaBoostRegressor(random_state=random_seed),
                "params": {
                    "n_estimators": [50],
                    "learning_rate": [0.1]
                }
            },
            "K-Nearest Neighbors Regressor": {
                "model": KNeighborsRegressor(),
                "params": {
                    "n_neighbors": [3],
                    "weights": ["uniform"]
                }
            },
            "Lasso Regression": {
                "model": Lasso(random_state=random_seed),
                "params": {
                    "alpha": [0.1]
                }
            },
            "Ridge Regression": {
                "model": Ridge(random_state=random_seed),
                "params": {
                    "alpha": [1.0]
                }
            },
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}  # Linear Regression has no hyperparameters
            },
            "MLP Regressor": {
                "model": MLPRegressor(random_state=random_seed),
                "params": {
                    "hidden_layer_sizes": [(50,)],
                    "activation": ["relu"],
                    "max_iter": [200]
                }
            },
            "Random Forest Regressor": {
                "model": RandomForestRegressor(random_state=random_seed),
                "params": {
                    "n_estimators": [50],
                    "max_depth": [3],
                    "min_samples_split": [2]
                }
            },
            "Support Vector Regressor (SVR)": {
                "model": SVR(),
                "params": {
                    "C": [1],
                    "kernel": ["linear"],
                    "epsilon": [0.1]
                }
            },
        }

        # Hyperparameter tuning and evaluation
        tuned_results = []

        for model_name in selected_models:
            st.write(f"Tuning ({model_name})...")
            details = models[model_name]
            
            if sampling_technique == "Train-Test Split":
                # Train-Test Split Evaluation
                random_search = RandomizedSearchCV(
                    estimator=details["model"],
                    param_distributions=details["params"],
                    n_iter=3,
                    cv=2,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    random_state=random_seed
                )
                random_search.fit(X_train, Y_train.values.ravel())
                best_model = random_search.best_estimator_
                predictions = best_model.predict(X_test)
                best_mae = np.abs(predictions - Y_test.values.ravel()).mean()
            else:
            # K-Fold Cross-Validation slider with unique key for each model
                n_folds = st.sidebar.slider(f"Number of Folds (K-Fold) for {model_name}", 2, 10, 5, key=f"kfold_slider_{model_name}")
                random_search = RandomizedSearchCV(
                    estimator=details["model"],
                    param_distributions=details["params"],
                    n_iter=3,
                    cv=n_folds,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    random_state=random_seed
                )
                random_search.fit(X, Y.values.ravel())
                best_model = random_search.best_estimator_
                best_mae = -random_search.best_score_

            tuned_results.append({
                "Model": model_name,
                "Best Parameters": random_search.best_params_,
                "Mean Absolute Error (MAE)": round(best_mae, 3)
            })

        # Display results
        st.write("### Tuned Model Performance Comparison")
        tuned_results_df = pd.DataFrame(tuned_results).sort_values(by="Mean Absolute Error (MAE)", ascending=True)
        st.dataframe(tuned_results_df)

        # Plot results as a horizontal bar chart
        st.write("### Tuned Model Mean Absolute Error (MAE) Bar Chart")
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(tuned_results_df)))
        plt.barh(
            tuned_results_df["Model"],  # Model names for the y-axis
            tuned_results_df["Mean Absolute Error (MAE)"],  # MAE values for the x-axis
            color=colors
        )
        plt.xlabel("Mean Absolute Error (MAE)")
        plt.ylabel("Model")
        plt.title("Tuned Model Performance Comparison")
        plt.gca().invert_yaxis()  # Invert the y-axis for better readability
        st.pyplot(plt)

        # Line chart for MAE
        st.write("### Mean Absolute Error (MAE) Line Chart")
        plt.figure(figsize=(10, 6))
        plt.plot(
            list(tuned_results_df["Model"]),  # Convert the "Model" column to a list of strings
            tuned_results_df["Mean Absolute Error (MAE)"].to_numpy(),  # Convert the MAE column to a numpy array
            marker='o',
            linestyle='-',
            label="MAE"
        )
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Model")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("Tuned Model Performance")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Best Model Conclusion
        if not tuned_results_df.empty:
            best_model = tuned_results_df.iloc[0]
            st.write("### Conclusion")
            st.write(
                f"The **best-performing model** is **{best_model['Model']}** with a Mean Absolute Error (MAE) "
                f"of **{best_model['Mean Absolute Error (MAE)']}**. The best hyperparameters are:"
            )
            # Directly display the dictionary of best parameters
            st.json(best_model["Best Parameters"])
            st.success("Hyperparameter Tuning Complete!")
        else:
            st.error("No results available. Please adjust settings and re-run.")

    else:
        st.error(f"Target column '{target_column}' not found in the dataset.")
else:
    st.write("Please upload a dataset.")
