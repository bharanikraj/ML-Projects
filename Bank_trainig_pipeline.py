from New_project import Bannking_EDA
from New_project_Preprocessing import Banking_preprocessing
from New_project_model_training import Banking_Model_Trainer
import pandas as pd

data = pd.read_csv('bank-full.csv')

eda = Bannking_EDA(data,target_col='y')
eda.detect_col_types().analzing_target_distribution()
eda.summary_stats()
eda.plot_numerical_dist()
eda.plot_categorical_dist()
eda.analyzing_correlation()
eda.detect_outliers()
eda.full_analysis()

preprocessor = Banking_preprocessing(target_col='y',test_size=0.2)
preprocessor.fit(data)

trainer = Banking_Model_Trainer(preprocessor=preprocessor)
trainer.tune_hyperparamters(preprocessor.X_train,preprocessor.Y_train)

results = trainer.evaluate(preprocessor.X_test,preprocessor.Y_test)
print(results['Random Forest']['test roc_auc'])