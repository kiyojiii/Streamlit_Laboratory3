import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import os
import joblib
import numpy as np

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Train Model", "Temperature Predictor"])

# Predefined CSV path
csv_path = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\csv\weather.csv"

if selection == "Train Model":
    st.title("Regression Model Training and Temperature Prediction")

    # Load dataset
    if os.path.exists(csv_path):
        dataframe = read_csv(csv_path)
        st.success("Dataset successfully loaded from predefined path.")
    else:
        st.error(f"Dataset not found at: {csv_path}")
        st.stop()

    # Show Raw Data Preview
    st.write("### Raw Dataset Preview")
    with st.expander("Show Raw Dataset", expanded=False):
        st.dataframe(dataframe)

    # Show Preprocessed Data
    st.write("### Preprocessed Dataset Preview")
    with st.expander("Show Preprocessed Dataset", expanded=True):
        st.dataframe(dataframe)

    # Preprocessing non-numeric columns
    description_mapping = {'Cold': 0, 'Warm': 1, 'Normal': 2}

    if 'Description' in dataframe.columns:
        dataframe['Description'] = dataframe['Description'].map(description_mapping)

    non_numeric_columns = dataframe.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_columns:
        for col in non_numeric_columns:
            if col != 'Description':
                dataframe[col] = dataframe[col].astype("category").cat.codes

    # Target and features
    target_column = "Temperature_c"
    if target_column not in dataframe.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        st.stop()

    feature_columns = [col for col in dataframe.columns if col != target_column]
    X = dataframe[feature_columns].values
    Y = dataframe[target_column].values

    # Quick Mode Toggle
    st.write("### Enable Quick Training")
    quick_mode = st.checkbox("Enable Quick Mode (Reduces Training Time)", value=True)
    if quick_mode:
        sample_size = st.slider("Sample Size (Percentage of Data)", 10, 100, 50)
        X, Y = X[:int(len(X) * sample_size / 100)], Y[:int(len(Y) * sample_size / 100)]
        st.warning(f"Training with {len(X)} samples (Quick Mode enabled).")

    # Model selection mapping
    model_mapping = {
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "ElasticNet": ElasticNet,
        "AdaBoostRegressor": AdaBoostRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "Lasso": Lasso,
        "Ridge": Ridge,
        "LinearRegression": LinearRegression,
        "MLPRegressor": MLPRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "SVR": SVR
    }

    # User-friendly dropdown for model selection
    model_choice = st.selectbox(
        "Select Regression Model",
        list(model_mapping.keys())  # Display clean model names
    )

    # Hyperparameter configurations
    if model_choice == "DecisionTreeRegressor":
        max_depth = st.slider("Max Depth", 1, 50, 10)
        model = model_mapping[model_choice](max_depth=max_depth, random_state=42)
    elif model_choice == "ElasticNet":
        alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        model = model_mapping[model_choice](alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    elif model_choice == "AdaBoostRegressor":
        n_estimators = st.slider("Number of Estimators", 10, 500, 50)
        model = model_mapping[model_choice](n_estimators=n_estimators, random_state=42)
    elif model_choice == "KNeighborsRegressor":
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
        model = model_mapping[model_choice](n_neighbors=n_neighbors)
    elif model_choice == "Lasso":
        alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
        model = model_mapping[model_choice](alpha=alpha, random_state=42)
    elif model_choice == "Ridge":
        alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
        model = model_mapping[model_choice](alpha=alpha, random_state=42)
    elif model_choice == "LinearRegression":
        model = model_mapping[model_choice]()
    elif model_choice == "MLPRegressor":
        hidden_layer_sizes = st.slider("Hidden Layer Size", 10, 200, 50)
        max_iter = st.slider("Max Iterations", 100, 1000, 200)
        model = model_mapping[model_choice](hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, random_state=42)
    elif model_choice == "RandomForestRegressor":
        n_estimators = st.slider("Number of Trees", 10, 500, 100)
        model = model_mapping[model_choice](n_estimators=n_estimators, random_state=42)
    elif model_choice == "SVR":
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = model_mapping[model_choice](kernel=kernel, C=1.0, epsilon=0.2)

    # Sampling technique
    sampling_technique = st.radio("Choose Sampling Technique:", ["Train-Test Split", "K-Fold Cross-Validation"])

    # Train-Test Split
    if sampling_technique == "Train-Test Split":
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        model.fit(X_train, Y_train)
        mae = np.abs(model.predict(X_test) - Y_test).mean()
        st.write(f"Mean Absolute Error (Test Set): {mae:.3f}")

    # K-Fold Cross-Validation
    elif sampling_technique == "K-Fold Cross-Validation":
        k = st.slider("Number of Folds (K)", 2, 10, 5)
        mae = -cross_val_score(model, X, Y, scoring="neg_mean_absolute_error", cv=KFold(n_splits=k)).mean()
        st.write(f"Mean Absolute Error (K-Fold CV): {mae:.3f}")

        # Explicitly fit the model after cross-validation
        model.fit(X, Y)

    # Save the model
    if st.button("Save Model"):
        save_folder = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\Models\regression"
        os.makedirs(save_folder, exist_ok=True)
        model_filename = os.path.join(save_folder, f"{sampling_technique}_{model_choice.replace(' ', '_')}_{mae}_model.joblib")
        joblib.dump(model, model_filename)
        st.success(f"Model saved successfully at: {model_filename}")

elif selection == "Temperature Predictor":
    st.header("Temperature Predictor")

    # Models directory
    models_directory = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\Models\regression"

    # Check if models exist in the directory
    if os.path.exists(models_directory):
        model_files = [file for file in os.listdir(models_directory) if file.endswith(".joblib")]
        
        if model_files:
            # Dropdown to select model
            selected_model = st.selectbox("Select a trained model:", model_files)

            # Load the selected model
            model_path = os.path.join(models_directory, selected_model)
            model = joblib.load(model_path)
            st.success(f"Model '{selected_model}' loaded successfully!")

            # Input form for new predictions
            st.subheader("Enter Feature Values to Predict the Temperature")

            # Create columns for input fields, two per row
            col1, col2 = st.columns(2)
            with col1:
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
                wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0)
            with col2:
                wind_bearing = st.number_input("Wind Bearing (degrees)", min_value=0, max_value=360, value=180)
                visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=20.0, value=10.0)

            col3, col4 = st.columns(2)
            with col3:
                pressure = st.number_input("Pressure (millibars)", min_value=900, max_value=1040, value=1010)
                rain = st.selectbox("Was / Is it Raining?", ["Yes", "No"])
            with col4:
                description = st.selectbox("How can you describe the weather?", ['Cold', 'Warm', 'Normal'])

            # Encode categorical inputs
            description_encoded = {'Cold': 0, 'Warm': 1, 'Normal': 2}[description]
            rain_encoded = {'Yes': 1, 'No': 0}[rain]

            # Prepare input data
            input_data = np.array([
                humidity, wind_speed, wind_bearing, visibility, pressure, rain_encoded, description_encoded
            ]).reshape(1, -1)

            # Perform the prediction
            if st.button("Predict Temperature"):
                try:
                    prediction = model.predict(input_data)
                    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("No models found in the directory. Please ensure models are saved in the correct folder.")
    else:
        st.error(f"Models directory '{models_directory}' not found. Please check the path.")

else:
    st.warning("Invalid selection. Please use the sidebar to navigate.")