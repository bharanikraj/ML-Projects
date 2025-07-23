import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder   
from New_project import Bannking_EDA

class Banking_preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, target_col='y', test_size=0.2, random_state=24):
        """
        Initializing preprocessing Pipeline

        target_col = name of target_column (By default is 'y'. So here also used)
        test_size = Proportion of the test split
        random_state = random seed (This is for reproducibility)
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
        self.scaler = StandardScaler()
        self.encoders = {}  # This will store the encoders for each categorical features
        self.ohe_feature_names = []  # Store feature names after OHE

    def type_based_col_split(self, df):
        """Here my plan is to bring the Bannking_eda class to split dataset based on dtype"""
        from New_project import Bannking_EDA  # Imported here # Otherwise it will be in circular imports
        self.eda = Bannking_EDA(df, self.target_col)
        self.eda.detect_col_types()     # Used Bannking_Eda's method

        # Below list comprehension is to exclude 'target_col' from encoding.
        self.numerical_cols = [col for col in self.eda.numerical_cols if col != self.target_col]
        self.categorical_cols = [col for col in self.eda.categorcal_cols if col != self.target_col]  
        
        return df

    def handling_nulls_and_duplicates(self, df):
        """Will handle nulls and duplicates if it is there
        
        For Replacing here below methods are used

        Median for Numerics
        Mode for Categoricals
        """
        df = df.copy()  # Make a copy to avoid modifying original
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col in self.numerical_cols:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)  # Fixed: mode() returns Series

        # Remove duplicates
        if df.duplicated().sum() > 0:
            df.drop_duplicates(inplace=True)
            print(f'Removed {df.duplicated().sum()} Duplicates')

        return df

    def splitting_dataset(self, df):
        """ Splitting dataset here to prevent data leakage """
        X = df.drop(self.target_col, axis=1)
        Y = df[self.target_col]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y
        )

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def handling_outliers(self, X):
        """Handle outliers in numerical features"""
        X = X.copy()
        cols_to_process = [col for col in self.numerical_cols if col in X.columns]
    
        for col in cols_to_process:
            # Calculate bounds
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Calculate outlier percentage
            outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
            outlier_percentage = len(outliers) / len(X) * 100

            # Handle outliers
            if outlier_percentage < 5:
                X[col] = X[col].clip(lower_bound, upper_bound)
            elif 5 <= outlier_percentage < 20:
                upper = X[col].quantile(0.95)
                lower = X[col].quantile(0.05)
                X[col] = X[col].clip(lower, upper)
            elif outlier_percentage >= 20:
                X[f"log_{col}"] = np.log1p(X[col])
                if col in self.numerical_cols:
                    self.numerical_cols.remove(col)
                    self.numerical_cols.append(f"log_{col}")
                X = X.drop(col, axis=1)
    
        return X

    def normalize_numerics(self, X, train=True):
        """Numerical feature Normalization"""
        X = X.copy()
        existing_numerical_cols = [col for col in self.numerical_cols if col in X.columns]

        if existing_numerical_cols:
            if train:
                X[existing_numerical_cols] = self.scaler.fit_transform(X[existing_numerical_cols])
            else:
                X[existing_numerical_cols] = self.scaler.transform(X[existing_numerical_cols])

        return X

    def encoding_categoricals(self, X, train=True):
        """Encoding categorical features based on label count (cardinality)"""
        X = X.copy()
        
        for col in self.categorical_cols:
            if col not in X.columns:
                continue

            if train:
                labels = X[col].nunique()
                if labels == 2:
                    encoder = LabelEncoder()
                    self.encoders[col] = encoder
                    X[col] = encoder.fit_transform(X[col])
                    print(f"Label encoding for {col} is done")
                
                elif labels > 2:
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    self.encoders[col] = encoder
                    
                    # Fit and transform
                    encoded_data = encoder.fit_transform(X[[col]])
                    
                    # Create feature names
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                    
                    # Create DataFrame with encoded features
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                    
                    # Drop original column and concatenate encoded features
                    X = X.drop(col, axis=1)
                    X = pd.concat([X, encoded_df], axis=1)
                    
                    print(f"OHE is completed for {col}")
                    
            else:  # For test/transform data
                if col in self.encoders:
                    encoder = self.encoders[col]
                    
                    if isinstance(encoder, LabelEncoder):
                        # Handle unknown labels in test data
                        X[col] = X[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                    
                    elif isinstance(encoder, OneHotEncoder):
                        # Transform using fitted encoder
                        encoded_data = encoder.transform(X[[col]])
                        
                        # Create feature names (same as training)
                        feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                        
                        # Create DataFrame with encoded features
                        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                        
                        # Drop original column and concatenate encoded features
                        X = X.drop(col, axis=1)
                        X = pd.concat([X, encoded_df], axis=1)

        return X
    
    def fit(self, df):
        """Fitting the preprocessing pipe
        
        This will learn the preprocessing rules from Training data"""
        
        self.type_based_col_split(df)
        print('Column split done')
    
        df = self.handling_nulls_and_duplicates(df)
        print("Nulls and duplicates are handled")
        
        self.splitting_dataset(df)
        print('Dataset splitted successfully')

        self.X_train = self.handling_outliers(self.X_train)
        print("Outlier handling completed")

        self.X_train = self.normalize_numerics(self.X_train, train=True)
        print("Numerical normalization completed")

        self.X_train = self.encoding_categoricals(self.X_train, train=True)
        print("Categorical Encoding is completed")

        return self
    
    def transform(self, df):
        """Transforming fitted transformations"""
        
        df = self.handling_nulls_and_duplicates(df)
        print("Nulls and duplicates are handled well")

        X = df.drop(self.target_col, axis=1)
        Y = df[self.target_col] if self.target_col in df else None
        print("Separation of target_col is also done")

        X = self.handling_outliers(X)
        print("Outlier handling is also completed")

        X = self.normalize_numerics(X, train=False)
        print("Numerical normalization completed")

        X = self.encoding_categoricals(X, train=False)
        print("Categorical encoding completed")

        return X, Y

    def fit_transform(self, df):
        """Convenience method to fit and transform in one go"""
        self.fit(df)
        return self.transform(df)