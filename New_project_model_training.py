from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd


class Banking_Model_Trainer():
    
    def __init__(self, preprocessor=None, tree_preprocessor=None):
        """
        Preprocessor - for non tree based models - uses Scaling
        tree_preprocessor - for tree based models - not uses scaling
        """
        self.preprocessor = preprocessor
        self.tree_preprocessor = tree_preprocessor
        self.results = {}  # Initialize results dictionary
        
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(class_weight='balanced', random_state=24),
                'params': {
                    'C': [0.01, 0.1, 1, 10],  
                    'penalty': ['l2']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(class_weight='balanced', random_state=24),
                'params': {
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5] 
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(class_weight='balanced', random_state=24),
                'params': {
                    'n_estimators': [50, 100],
                    'max_features': ['sqrt', 'log2']
                }
            # },
            # 'SVC': {
            #     'model': SVC(class_weight='balanced', probability=True, random_state=24),
            #     'params': {
            #         'C': [0.1, 1, 10],
            #         'kernel': ['linear', 'rbf']
                # }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            }
        }

    def get_preprocessor(self, model_name):
        """Selecting Preprocessor based on model type
        because tree models preprocessing is different"""
        return self.tree_preprocessor if ('Tree' in model_name or 'Forest' in model_name) else self.preprocessor
    
    def tune_hyperparamters(self, X_train, Y_train, cv=5):
        """Performing Hyperparameter tuning using Grid_search CV"""
        print("Starting hyperparameter tuning...")
        
        for name, config in self.models.items():
            print(f"\nTuning {name}...")
            
            preprocessor = self.get_preprocessor(name)
            
            # Use the already processed training data from preprocessor
            if preprocessor:
                X_processed = X_train  # X_train is already processed
            else:
                X_processed = X_train
            
            # GridSearch CV
            grid = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid.fit(X_processed, Y_train)
            
            # Store best models and results
            self.results[name] = {
                'best_model': grid.best_estimator_,
                'best_params': grid.best_params_,
                'best_score': grid.best_score_,
                'preprocessor': preprocessor
            }
            
            print(f"{name} - Best Score: {grid.best_score_:.4f}")
        
        return self.results
    
    def evaluate(self, X_test, Y_test):
        """Evaluating tuned models on test set"""
        evaluation = {}
        
        for name, data in self.results.items():
            print(f"\nEvaluating {name}...")
            
            model = data['best_model']
            preprocessor = data['preprocessor']
            
            # Use the already processed test data
            X_processed = X_test  # X_test should already be processed
            
            y_pred = model.predict(X_processed)
            y_proba = model.predict_proba(X_processed)[:,1] if hasattr(model, 'predict_proba') else None
            
            evaluation[name] = {
                'best_params': data['best_params'],
                'test_roc_auc': roc_auc_score(Y_test, y_proba) if y_proba is not None else None,
                'classification_report': classification_report(Y_test, y_pred)
            }
        
        return evaluation
    
    def get_model_summary(self):
        """Display model training summary"""
        print("\n" + "="*50)
        print("MODEL TRAINING SUMMARY")
        print("="*50)
        
        for name, data in self.results.items():
            print(f"\n{name}:")
            print(f"  Best CV Score: {data['best_score']:.4f}")
            print(f"  Best Parameters: {data['best_params']}")
    
    def get_best_model(self):
        """Get the best performing model based on CV score"""
        if not self.results:
            raise ValueError("No models have been trained yet!")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['best_score'])
        best_model_data = self.results[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best CV Score: {best_model_data['best_score']:.4f}")
        
        return best_model_name, best_model_data
    
    def predict(self, model_name, X_new):
        """Making predictions with the best tuned models"""
        model_data = self.results.get(model_name)
        if not model_data:
            raise ValueError(f'{model_name} not found in trained models')
        
        model = model_data['best_model']
        
        # X_new should already be processed if coming from the same preprocessor
        return model.predict(X_new)