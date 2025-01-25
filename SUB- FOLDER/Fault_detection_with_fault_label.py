import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


output_dir = r"11models_F20"
# output_dir = r"D:\MAJOR\Major-Project\SUB- FOLDER\models"
os.makedirs(output_dir, exist_ok=True)  



data = pd.read_csv("Training_Data_prev_F20.csv")
# data = pd.read_csv("Training_Data.csv")


data['Fault_label'] = data['Fault_label'].astype(str).str.strip()


label_encoder = LabelEncoder()
data['Fault_label'] = label_encoder.fit_transform(data['Fault_label'])


joblib.dump(label_encoder, os.path.join(output_dir, "fault_label_encoder.pkl"))


features = ['Inv kW Sum (kW)', 'Load kW Sum (kW)', 'Solar kW (kW)', 'Ambient Temp (C)', 'Solar Radiation (W/m2)', 
            'Inv Exp kWh', 'Inv Imp kWh', 'Src A Exp kWh', 'Src A Imp kWh', 'Src B Exp kWh', 'Src B Imp kWh', 
            'Site kWh (calc)', 'Batt Exp kWh', 'Batt Imp kWh', 'Solar kWh']
X = data[features]
y = data['Fault_label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB()
}


combined_results = X_test.copy()
combined_results['Actual_Fault_Label'] = y_test.values
combined_results['Actual_Fault_Label'] = label_encoder.inverse_transform(combined_results['Actual_Fault_Label'])


# Train, evaluate and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save the trained model
    model_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model, model_path)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    

    combined_results[f'{name}_Pred_Fault_Label'] = y_pred
    combined_results[f'{name}_Pred_Fault_Label'] = label_encoder.inverse_transform(combined_results[f'{name}_Pred_Fault_Label'])


output_path = os.path.join(output_dir, "all_models_predictions.csv")
combined_results.to_csv(output_path, index=False) 
print(f"\nCombined results saved to '{output_path}'")

# Test on new data function
def predict_new_data(file_path, model_name):
    # Load new data
    new_data = pd.read_csv(file_path)


    new_data_processed = new_data[features]  
    

    model_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    loaded_model = joblib.load(model_path)
    encoder_path = os.path.join(output_dir, "fault_label_encoder.pkl")
    loaded_encoder = joblib.load(encoder_path)
    

    new_data['Pred_Fault_Label'] = loaded_model.predict(new_data_processed)
    new_data['Pred_Fault_Label'] = loaded_encoder.inverse_transform(new_data['Pred_Fault_Label'])
    

    output_file = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_new_data_predictions.csv")
    new_data.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'")
    

