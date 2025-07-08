from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score ,classification_report
from sklearn.model_selection import cross_val_score,GridSearchCV
import pandas as pd


class Banking_Model_Trainer():
    
    def __init__(self,preprocessor=None, tree_preprocessor=None):
        """
        Preprocessor - for non tree based models - uses Scaling
        tree_preprocessor - for tree based models - not uses scaling
        
        """

        self.preprocessor = preprocessor
        self.tree_preprocessor = tree_preprocessor
        self.models = {
                        'Logistic Regrssion' :{ 'model' :LogisticRegression(class_weight='balanced'),
                                               'params':
                                               { 'C':[0,0.1,0.1,1,10],'penlaty':['l2']}},
                        
                        'Decision Tree' : {'model': DecisionTreeClassifier(class_weight='balanced'),
                                           'params': {'max_depth':[3,5,7],
                                                      'min_sample_split':[2,5]}},
                        'Random Forest' : {'model': RandomForestClassifier(class_weight='balanced'),
                                           'params':{'n_estimators': [50,100],
                                                     'max_features':['sqrt','log2']}},
                        'SVC'           :{'model':SVC(class_weight='balanced',probability=True),
                                          'params':{'C':[0.1,1,10],'kernel':['linear','rbf']}},
                        'KNN'           :{'model':KNeighborsClassifier(),
                                          'params':{'n_neighbors':[3,5,7],
                                                    'weights':['uniform','distance']}}
                        
        }

    def get_preprocessor(self,model_name):
        """Selecting Preprocessor based on model type
        because tree models preprocessing is different"""
        return self.tree_preprocessor if ('Tree' in model_name or 'Forest' in model_name) else self.preprocessor()
    
    def tune_hyperparamters(self,Xtrain,Ytrain,cv=5):
        """Performing Hyperparameter tuning using Grid_search CV"""
        for name,config in self.models.items():
            preprocessor = self.get_preprocessor(name)
            X_processed = preprocessor.transform(Xtrain) if preprocessor else Xtrain

            ## GridSearch CV

            grid = GridSearchCV(estimator=config['model'],
                                param_grid=config['params'],
                                cv=cv,scoring='roc-auc',n_jobs=-1)
            grid.fit(X_processed,Ytrain)

            ## Below is to store best models and results
            self.results[name] = { 
                'best_model': grid.best_estimator_,
                'best_params': grid.best_params_,
                'best_score': grid.best_score_,
                'preprocessor':preprocessor
            }
        return self.results
    
    def evaluate(self,X_test,Y_test):
        """Evaluating tuned models on test set"""
        evaluation = {}

        for name,data in self.results.items():
            model = data['best_model']
            preprocessor = data['preprocessor']
            X_processed = preprocessor.transform(X_test) if preprocessor else X_test

            y_pred = model.predict(X_processed)
            y_proba = model.predict_proba(X_processed)[:,1] if hasattr(model,'predict_proba') else None

            evaluation[name] = {
                'best_params' : data['best_params'],
                'test_roc-auc':roc_auc_score(Y_test,y_proba) if y_proba else None,
                'classification_report' : classification_report(Y_test,y_pred)

            }
        return evaluation
    
    def predict(self, model_name,X_new):
        """Making predictions with the best tuned models"""

        model_data = self.results.get(model_name)
        if not model_data:
            raise ValueError(f'{model_name} not found')
        preprocessor = model_data['preprocessor']
        model = model_data['best_model']
        x_processed = preprocessor.transform(X_new) if preprocessor else X_new

        return model.predict(x_processed)