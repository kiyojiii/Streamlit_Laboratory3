import streamlit as st
from pandas import read_csv, get_dummies
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os
import joblib
import numpy as np

# Streamlit app title
st.title("Classification Model Training and Lung Cancer Prediction")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go To", ["Train Model", "Make Predictions"])

if section == "Train Model":
    st.header("Train a Classification Model")

    # Load the dataset from predefined path
    dataset_path = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\csv\survey lung cancer.csv"
    try:
        dataframe = read_csv(dataset_path)
        st.write("Dataset Preview:")
        with st.expander("Show Complete Dataset", expanded=False):
            st.dataframe(dataframe)

        # Convert columns from Smoking to Chest Pain (2 = 1, 1 = 0)
        columns_to_convert = [
            "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE",
            "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
            "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
        ]
        if all(col in dataframe.columns for col in columns_to_convert):
            for col in columns_to_convert:
                dataframe[col] = dataframe[col].replace({2: 1, 1: 0})
            st.write("Processed Dataset (Converted Smoking to Chest Pain Columns):")
            st.write(dataframe.head())
        else:
            missing_columns = [col for col in columns_to_convert if col not in dataframe.columns]
            st.warning(f"The following required columns are missing from the dataset: {missing_columns}")

        # One-Hot Encoding for categorical features
        dataframe = get_dummies(dataframe, drop_first=True)
        st.write("Processed Dataset (After Encoding):")
        st.write(dataframe.head())

        # Automatically set the target column
        columns = list(dataframe.columns)
        target_column = "LUNG_CANCER_YES" if "LUNG_CANCER_YES" in columns else None
        if target_column:
            st.success(f"Target column automatically set to: {target_column}")
        else:
            target_column = st.selectbox("Select the target column", columns)

        # Default feature columns
        default_features = [col for col in columns if col != target_column]
        feature_columns = st.multiselect(
            "Select feature columns",
            options=[col for col in columns if col != target_column],
            default=default_features
        )

        if target_column and feature_columns:
            X = dataframe[feature_columns].values
            Y = dataframe[target_column].values

            # Model selection
            model_choice = st.selectbox(
                "Select Machine Learning Model",
                [
                    "Decision Tree", "Gaussian Naive Bayes", "AdaBoost", "K-Nearest Neighbors",
                    "Logistic Regression", "MLP Classifier", "Perceptron Classifier", "Random Forest",
                    "Support Vector Machine (SVM)"
                ]
            )

            # Model initialization
            if model_choice == "Decision Tree":
                max_depth = st.slider("Max Depth", 1, 50, 5)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            elif model_choice == "Gaussian Naive Bayes":
                var_smoothing = st.number_input("Var Smoothing", 1e-9, 1e-5, 1e-8, format="%.1e")
                model = GaussianNB(var_smoothing=var_smoothing)

            elif model_choice == "AdaBoost":
                n_estimators = st.slider("Number of Estimators", 50, 500, 100)
                model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)

            elif model_choice == "K-Nearest Neighbors":
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)

            elif model_choice == "Logistic Regression":
                C = st.slider("Inverse of Regularization Strength (C)", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, max_iter=200, random_state=42)

            elif model_choice == "MLP Classifier":
                hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 100,50)", "100,50")
                hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(",")))
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=200, random_state=42)

            elif model_choice == "Perceptron Classifier":
                eta0 = st.slider("Learning Rate (eta0)", 0.01, 1.0, 0.1)
                model = Perceptron(eta0=eta0, random_state=42)

            elif model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            elif model_choice == "Support Vector Machine (SVM)":
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
                model = SVC(kernel=kernel, C=C, random_state=42)

            # Sampling technique selection
            sampling_technique = st.radio(
                "Choose Sampling Technique",
                ["Train-Test Sets", "K-Fold Cross Validation"]
            )

            # Training process
            if sampling_technique == "Train-Test Sets":
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
                if st.button("Train Model"):
                    model.fit(X_train, Y_train)
                    accuracy = model.score(X_test, Y_test)
                    st.session_state["trained_model"] = model
                    st.session_state["model_accuracy"] = accuracy
                    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            elif sampling_technique == "K-Fold Cross Validation":
                k = st.slider("Number of Folds (K)", 2, 10, 5)
                if st.button("Train Model with K-Fold"):
                    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X, Y, cv=kfold)
                    st.write(f"K-Fold Cross Validation Mean Accuracy: {np.mean(scores) * 100:.2f}%")
                    st.write(f"K-Fold Accuracy Standard Deviation: {np.std(scores) * 100:.2f}%")
                    # Retrain the model for saving
                    model.fit(X, Y)
                    st.session_state["trained_model"] = model
                    st.session_state["model_accuracy"] = np.mean(scores)

            # Save model button
            if "trained_model" in st.session_state and st.session_state["trained_model"]:
                if st.button("Save Model", key="save_model_button"):
                    save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\Models\classification'
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(
                        save_folder,
                        f"{sampling_technique}_classification_{model_choice.replace(' ', '_').lower()}_{st.session_state['model_accuracy'] * 100:.2f}_accuracy.joblib"
                    )
                    joblib.dump(st.session_state["trained_model"], save_path)
                    st.success(f"Model saved as: {save_path}")

    except Exception as e:
        st.error(f"Error loading the dataset: {e}")

elif section == "Make Predictions":
    st.header("Lung Cancer Diagnosis Predictor")

    # Predefined models directory
    models_directory = r"C:\Users\user\Desktop\jeah\ITD105\LABORATORY3\Models\classification"

    try:
        # List all available models in the directory
        available_models = [file for file in os.listdir(models_directory) if file.endswith(".joblib")]

        if available_models:
            # Model selection dropdown
            selected_model = st.selectbox("Select a Trained Model:", available_models)

            # Load the selected model
            model_path = os.path.join(models_directory, selected_model)
            model = joblib.load(model_path)
            st.success(f"Model '{selected_model}' successfully loaded!")

            st.subheader("Input Sample Data for Prediction")

            # Input Fields for Prediction
            def convert_yes_no(value):
                return 1 if value == "Yes" else 0  # Adjusted for the new mapping (Yes=1, No=0)

            # Input features (15 expected by the model), following the specified order
            # Row 1: Gender, Age, Smoking
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                gender = 1 if gender == "Male" else 0  # Convert "Male" to 1 and "Female" to 0
            with col2:
                age = st.number_input("Age", min_value=0, max_value=120, value=1)
            with col3:
                smoking = convert_yes_no(st.radio("Do you smoke?", ["Yes", "No"]))

            # Row 2: Anxiety, Alcohol, Chronic Disease, Chest Pain
            col1, col2, col3 = st.columns(3)
            with col1:
                anxiety = convert_yes_no(st.radio("Do you have anxiety?", ["Yes", "No"]))
            with col2:
                alcohol_consumption = convert_yes_no(st.radio("Do you consume alcohol?", ["Yes", "No"]))
            with col3:
                chronic_disease = convert_yes_no(st.radio("Do you have a chronic disease?", ["Yes", "No"]))

            # Row 3: Peer Pressure, Allergy, Wheezing, Fatigue
            col1, col2, col3 = st.columns(3)
            with col1:
                peer_pressure = convert_yes_no(st.radio("Are you under peer pressure?", ["Yes", "No"]))
            with col2:
                allergy = convert_yes_no(st.radio("Do you have any allergy?", ["Yes", "No"]))
            with col3:
                wheezing = convert_yes_no(st.radio("Do you experience wheezing?", ["Yes", "No"]))

            # Row 4: Coughing, Shortness of Breath, Difficulty Swallowing, Yellow Fingers
            col1, col2, col3 = st.columns(3)
            with col1:
                coughing = convert_yes_no(st.radio("Do you have a cough?", ["Yes", "No"]))
            with col2:
                shortness_of_breath = convert_yes_no(st.radio("Do you experience shortness of breath?", ["Yes", "No"]))
            with col3:
                swallowing_difficulty = convert_yes_no(st.radio("Do you have any difficulty swallowing?", ["Yes", "No"]))

            #Row 5
            col1, col2, col3 = st.columns(3)
            with col1:
                chest_pain = convert_yes_no(st.radio("Do you experience chest pain?", ["Yes", "No"]))
            with col2:
                fatigue = convert_yes_no(st.radio("Do you experience any fatigue?", ["Yes", "No"]))
            with col3:
                yellow_fingers = convert_yes_no(st.radio("Do you have yellow fingers?", ["Yes", "No"]))

            # Combine inputs
            input_data = [
                gender, age, smoking, anxiety, alcohol_consumption, chronic_disease, chest_pain,
                peer_pressure, allergy, wheezing, fatigue, coughing, shortness_of_breath,
                swallowing_difficulty, yellow_fingers
            ]

            # Predict using the loaded model
            if st.button("Predict"):
                prediction = model.predict([input_data])
                if prediction[0] == 1:
                    result = '<span style="color:green;">Positive for Lung Cancer</span>'
                else:
                    result = '<span style="color:red;">Negative for Lung Cancer</span>'
                
                st.markdown(f"### Prediction: {result}", unsafe_allow_html=True)

        else:
            st.warning("No trained models found in the directory. Please ensure models are available in the specified folder.")

    except Exception as e:
        st.error(f"Error accessing the models directory or loading the model: {e}")


else:
    st.warning("Invalid selection. Please use the sidebar to navigate.")
