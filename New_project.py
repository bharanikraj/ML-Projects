##  Importing dependencies for EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy import stats
import seaborn as sns


df = pd.read_csv("C:/Users/bharani/Downloads/bank+marketing/bank/bank-full.csv")


class Bannking_EDA:

    def __init__(self,df,target_col = 'y'):
        """ Initializing with the dataframe and target column 'y' """
        self.df = df.copy()
        self.target_col = target_col
        self.numerical_cols = None
        self.categorcal_cols = None


    def detect_col_types(self):
        """ Identifying the Numerical and Categorical cols """

        self.numerical_cols = self.df.select_dtypes(include = ['number']).columns.to_list()

        self.categorcal_cols = self.df.select_dtypes(exclude = ['number']).columns.to_list()

        if self.target_col in self.numerical_cols and self.df[self.target_col].nunique() <= 5:

            self.numerical_cols.remove(self.target_col)
            self.categorcal_cols.append(self.target_col)

        return self
    
    def summary_stats(self):
        """ Generating Comprehensive Statistics """

        print('Overview of the Dataset')
        print(f'shape : {self.df.shape}')
        print(f'\n Missing Values : \n {self.df.isnull().sum()}')

        if self.numerical_cols:
            print('\n Numerical Features Summary')
            print(self.df[self.numerical_cols].describe(percentiles = [0.1 , 0.25, 0.5, 0.75, 1.0]))

        if self.categorcal_cols:
            print('\n Categorical Features Summary')
            for col in self.categorcal_cols:
                print(f' \n {col} : ')
                print(self.df[col].value_counts(normalize = True))

        return self
    
    def analzing_target_distribution(self):
        """ Analyzing Target Value Distribution """

        if self.target_col in self.df.columns:
            print(f' Target Variable Analysis {self.target_col} :')
            target_dist = self.df[self.target_col].value_counts(normalize = True)
            print(target_dist)

            plt.figure(figsize=(8,4))
            sns.countplot(x=self.target_col,data=self.df)
            plt.title('Target Variable Distributiion')
            plt.show()

            ## Below is to understand the class imbalance ratio

            imbalance_ratio = target_dist.iloc[0] / target_dist.iloc[1]
            print(f' \n Class imbalance Ratio : {imbalance_ratio :.1f} :1')

        return self
    
    def plot_numerical_dist(self, cols = None):
        """ Plot Distributions of Numerical Features """
        cols = cols or self.numerical_cols
        if not cols:
            return self
        
        print('\n Numerical Features Distribution : ')

        for col in cols:
            plt.figure(figsize=(10,4))

            ## Histogram with KDE

            plt.subplot(1,2,2)
            sns.boxplot(y=self.df[col])
            plt.title(f' {col} Boxplot')
            plt.tight_layout()
            plt.show()

            ## skewness and Kurtosis
            skew = stats.skew(self.df[col].dropna())
            kurt = stats.kurtosis(self.df[col].dropna())
            print(f' {col} - Skewness : {skew:.2f},Kurtosis : {kurt:.2f}')

        return self
    
    def plot_categorical_dist(self,cols = None):
        """ Plot Distributions Of Categorical Features """
        cols = cols or self.categorcal_cols
        if not cols or self.target_col not in self.df.columns:
            return self
        
        print('\n Categorical Features Distribution :')

        for col in cols:
            if col == self.target_col:
                continue

            plt.figure(figsize=(12,5))

            ## Countplot

            plt.subplot(1,2,1)
            sns.countplot(x=col,data=self.df)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)

            ## Stacked Barplt vs Target
            plt.crosstab(self.df[col],self.df[self.target_col]).plot(kind = 'bar',stacked = True)
            plt.title(f'{col} vs Target')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        return self
    
    def analyzing_correlation(self):
        """Analysing Feature correlations"""

        if not self.numerical_cols or self.target_col not in self.df.columns:
            return self
        
        print('\n Correlation Analysis')

        corr_matrix = self.df[self.numerical_cols + [self.target_col]].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix,annot=True,cmap='RdBu',center=0)
        plt.title('Correlation of numerical fetaures')
        plt.show()


        ##  Top Correlation with target

        target_corr = corr_matrix[self.target_col].abs().sort_values(ascending = False)
        print('\n Top features correlated with target')
        print(target_corr[1:6]) # to skip the target featur itself 

        return self 
    
    def detect_outliers(self):
        """ Detect and Analyze Outliers in Numerics """
        if not self.numerical_cols:
            return self
        
        print('Outlier Detection :')
        outlier_results = {}

        for col in self.numerical_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_pct = (len(outliers ) / len(self.df)) * 100

            outlier_results[col] = {

                'outlier_count' : len(outliers),
                'oultier_percentage' : outlier_pct,
                'lower_bound ': lower_bound,
                'upper_bound' : upper_bound
            }

            print(f'{col}: {len(outliers)} outliers {outlier_pct:.2f}')

        return outlier_results
    

    def full_analysis(self):
        """
        Runng complete EDA
        """

        return (self.detect_col_types()
                .summary_stats()
                .analzing_target_distribution()
                .plot_numerical_dist()
                .plot_categorical_dist()
                .analyzing_correlation()
                .detect_outliers())
                




##          Now Initializing with df

eda = Bannking_EDA(df,target_col='y')

##          Calculating Full Analysis

# eda.full_analysis()

outlier_report = eda.detect_outliers()