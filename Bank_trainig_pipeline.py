from New_project import Bannking_EDA
from New_project_Preprocessing import Banking_preprocessing
from New_project_model_training import Banking_Model_Trainer
import pandas as pd

# Load data
data = pd.read_csv('bank-full.csv', delimiter=';', quotechar='"')

# EDA (optional)
eda = Bannking_EDA(data, target_col='y')
# eda.full_analysis()  # Uncomment if you want to run EDA

print('Preprocessing Starts Here')
preprocessor = Banking_preprocessing()
preprocessor.fit(data)

print(f'Training data shape: {preprocessor.X_train.shape}')
print(f'Test set data shape: {preprocessor.X_test.shape}')
print(f'Training data columns: {preprocessor.X_train.columns.tolist()}')

# Model Training
print('\nModel training starts here')
trainer = Banking_Model_Trainer(preprocessor=preprocessor)

# Tuning Hyperparameters
results = trainer.tune_hyperparamters(preprocessor.X_train, preprocessor.Y_train)

# Display Training Summary
trainer.get_model_summary()

# Test set evaluation
print('\nEvaluating models on test set')
evaluation_results = trainer.evaluate(preprocessor.X_test, preprocessor.Y_test)

print('\n' + '*'*50)
print('DISPLAYING RESULTS')
print('*'*50)

for model_name, metrics in evaluation_results.items():
    print(f'\n{model_name}:')
    print(f'Test ROC-AUC: {metrics["test_roc_auc"]:.3f}')
    print(f'Best Parameters: {metrics["best_params"]}')
    print('Classification Report:')
    print(metrics["classification_report"])

# Getting the best model
best_model_name, best_model_data = trainer.get_best_model()

# Example of making predictions on new data
print(f'\nMaking Predictions with {best_model_name}')
sample_prediction = trainer.predict(best_model_name, preprocessor.X_test.head(5))
print(f'Sample predictions: {sample_prediction}')