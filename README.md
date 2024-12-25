CSV Folder: Contains CSV Files Classification - survey lung cancer Regression - weather

Models Folder: Contains Classification and Regression Folders

Pages Folder: Contains separate py files for the activity: 1 - Classification Model Training, 2 - Classification Model Comparison, 3 - Classification model Hypertuning, 4 - Linear Model Training, 5 - Linear Model Comparison, 6 - Linear Model Hypertuning

Features: Dashboard

Displays Insightful data for the CSV Files
Displays a summary of graphs for each CSV File
Overview and a sample of what the CSV Files contain

1 - Classification Model Training
* Trains Models using different algorithms with sampling technique
* Trains Models using the survey lung cancer CSV file, the user can choose to train using K-Fold or Split into Train-Test Sets
* User can choose / pick K-Fold number of Folds, User can choose / pick test size for Train-Test
* Lung Cancer Diagnosis Predictor - Predicts if a person has lung cancer based on the questions / symptoms
* Target Column = "LUNG_CANCER" (What the model Predicts)
* Feature Column = "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN" (What the prediction will base off)

2 - Classification Model Comparison
* Compares different machine learning algorithms with sampling technique
* Displays in a table, bar, line and heatmap charts
* Uses performance metric "Classification Accuracy" to compare algorithm effectiveness
* Classification Accuracy: Measures the percentage of correct predictions made by a model. 

3 - Classification model Hypertuning
* Trains the CSV File to find what algorithm, sampling technique, parameters will give the best model for classification
* Only suggests the best parameters for the best model outcome

4 - Linear Model Training
* Trains Models using different algorithms with sampling technique
* Trains Models using the weather CSV file, the user can choose to train using K-Fold or Split into Train-Test Sets
* User can choose / pick K-Fold number of Folds, User can choose / pick test size for Train-Test 
* Temperature Predictor - Predicts the temperature based on the questions
* Target Column = "Temperature_c" (What the model Predicts)
* Feature Column = "Humidty" "Wind_Speed_kmh" "Wind_Bearing_degrees" "Visibility_km" "Pressure_millibars" "Rain" "Description" (What the prediction will base off)

5 - Linear Model Comparison
* Compares different machine learning algorithms with sampling technique
* Displays in a table, bar, line and heatmap charts
* Uses performance metric "Mean Absolute Error (MAE)" to compare algorithm effectiveness
* MAE (Mean Absolute Error): Measures the average absolute difference between predicted values and actual values, providing a straightforward interpretation of error magnitude.

6 - Linear Model Hypertuning
* Trains the CSV File to find what algorithm, sampling technique, parameters will give the best model for regression
* Only suggests the best parameters for the best model outcome

How to use: 2 Methods 
1 - Clone Method
create a Fresh New Folder on your Desktop
open that folder (make sure the directory is of that folder) on VS Code
type on the terminal of the VS Code "git clone https://github.com/kiyojiii/Streamlit_Laboratory3"
If all goes well, you should have a clone of this repository

2 - Manual Method
create a Fresh New Folder on your Desktop
go to this repository "https://github.com/kiyojiii/Streamlit_Laboratory3"
click on the green button that says "<> Code"
wait for the zip to download, extract the zip on your Fresh New Folder
After having the files on the folder, just type on the terminal "streamlit run Dashboard.py"

If you have complete packages, then you should have no errors and the streamlit app will run smoothly P.S don't forget to change the save model, csv file paths on the py files
