import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder   
from New_project import Bannking_EDA

class Banking_preprocessing(BaseEstimator,TransformerMixin):

    def __init__(self,target_col='y',test_size=0.2,random_state=24):
        """
        Initializing preprocessing Pipeline

        taregt_col = name of target_column (By default is 'y'. So here also used)
        test_size = Proportion of the test split
        random_state = random seed (This is for reproducubility)
        """

        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.numerical_cols = []
        self.categorical_cols = []
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.scalar = StandardScaler()

        self.encoders = {} ## This will store the encoders for each categorical features



    def type_based_col_split(self,df):
        """Here my plan is to bring the Bannking_eda class to split dataset based on dtype"""
        from New_project import Bannking_EDA # Imported here # Otherwise it will be in circular imports
        self.eda = Bannking_EDA(df,self.target_col)

        self.eda.detect_col_types()     ## Used Bannking_Eda's method

        ## Below list comprehension is to exclude 'target_col' from encoding.
        self.numerical_cols = [col for col in self.eda.numerical_cols if col != self.target_col]
        self.categorical_cols = [col for col in self.eda.categorcal_cols if col != self.target_col]  
        

        return df
    

    def handling_nulls_and_duplicates(self,df):
        """Will handle nulls and duplicates if it is there
        
        For Replacing here below methods are used

        Median for Numerics
        Mode for Categoricals
        """
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col in self.numerical_cols:
                    df[col].fillna(df[col].median(),inplace=True)

                else:
                    df[col].fillna(df[col].mode(),inplace= True)

        for col in df.columns:
            if df[col].duplicated().sum() > 0:
                df.drop_duplicates(inplace = True)

                print(f'Removed {df.duplicated().sum()} Duplicates')

        return df
    def splitting_dataset(self,df):

        """ Splitting dataset here to prevent data leakage """
        X = df.drop(self.target_col,axis=1)
        Y = df[self.target_col]



        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X,Y,test_size=self.test_size,random_state=self.random_state,stratify=Y)

        return self.X_train,self.X_test,self.Y_train,self.Y_test

    

    def handling_outliers(self,X):
        """First count the number of outliers to understand which technique is good to handle the outliers"""

        for col in self.numerical_cols:
            ## Now to check the original or log transformed column name

            current_col = f"log_{col}" if f"log_{col}" in X.columns else col

            if current_col not in X.columns:
                continue

            
            q1 = X[col].quantile(0.25)
            q3= X[col].quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr

            upper_bound = q3 + 1.5 * iqr

            outliers = X[( X[col] < lower_bound ) | ( X[col] > upper_bound )]
            outlier_percentage = len(outliers) / len(X) * 100


        ##  Now handle outliers based on the count

            if outlier_percentage < 5:
                X[col] = X[col].clip(lower_bound,upper_bound)

            elif 5 >= outlier_percentage < 20:
                upper = X[col].quantile(0.95)
                lower = X[col].quantile(0.05)

                X[col] = X[col].clip(lower,upper)

            else:
                if outlier_percentage > 20:
                    X[col] = np.log1p(X[col])
                    X.rename(columns = {col:f"log_{col}"},inplace=True)

            return X
                
    def normalize_numerics(self,X):
        """Numerical feature Normalization"""

        if len(self.numerical_cols) > 0:
            self.numerical_cols = self.scalar.fit_transform(X[self.numerical_cols])

        return X

    def encoding_categoricals(self,X,train = True):
        """Encoding categorical features based on label count (cardinality)
        
        Here used train = True - to make a if condition for encoding"""

        for col in self.categorical_cols:
            labels = X[col].nunique()

            if labels == 2:
                Encoder = LabelEncoder()

                if train:
                    self.encoders[col] = Encoder.fit(X[[col]])

                encoded = self.encoders[col].transform(X[[col]])
                X[col] = encoded
                print(f"Label encoding for {col} is done")
                

            elif labels > 2:
                """ Only OHE is used, for Logistic Regression
                    For other classification models like tree baesd, have to use other encoders
                """
                Encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

                if train:
                    self.encoders[col] = Encoder.fit(X[[col]])
                    X[col] = self.encoders[col].transform(X[[col]])
                    print(f"OHE is completed for {col}")

        return X
    
    def fit(self,df):
        """Fitting the preprocessing pipe
        
        This will learn the preprocessing rules from Training data"""

        self.type_based_col_split(df)
        print('Column split done')
    
        self.handling_nulls_and_duplicates(df)
        print("Nulls and duplicates are handled")
        
        self.splitting_dataset(df)
        print('Dataset splitted successfully')

        self.X_train = self.handling_outliers(self.X_train)

        self.X_train = self.normalize_numerics(self.X_train)

        self.X_train = self.encoding_categoricals(self.X_train,train=True)
        print("Categorical Encoding is completed")

        return self
    
    def transform(self,df):
        """Transforming fitted transformations"""

        df = self.handling_nulls_and_duplicates(df)
        print("Nulls and duplicated are handled well")

        X = df.drop(self.target_col,axis=1)
        Y = df[self.target_col] if self.target_col in df else None
        print("Seperation of target_col is also done")

        X = self.handling_outliers(X)
        print("Outlier handling is also completed")

        if len(self.numerical_cols) > 0:
            X[self.numerical_cols] = self.scalar.transform(X[self.numerical_cols])

        X = self.encoding_categoricals(X,train=False)

        return X, Y
    
    

data = pd.read_csv('bank-full.csv',delimiter=';',quotechar='"')

preprocessing = Banking_preprocessing()

preprocessing.fit(data)
preprocessing.transform(data)
