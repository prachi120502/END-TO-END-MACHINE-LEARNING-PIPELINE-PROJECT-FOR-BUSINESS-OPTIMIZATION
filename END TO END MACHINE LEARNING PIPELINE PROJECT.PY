import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Data Collection
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Data Preprocessing
def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Encode categorical variables if necessary
    data = pd.get_dummies(data)
    
    # Split the dataset
    X = data.drop("target", axis=1)  # Replace 'target' with your target variable
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Step 3: Model Training
def train_model(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

# Step 5: Model Deployment
def save_model(model, file_name):
    joblib.dump(model, file_name)

# Main function to run the pipeline
if __name__ == "__main__":
    # Load data
    data = load_data("data.csv")  # Replace with your dataset path
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, "business_optimization_model.pkl")