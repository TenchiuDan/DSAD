import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.simplefilter(action='ignore', category=FutureWarning)

# print("Helo")

# np.random.seed(0)
# data = np.random.rand(50, 6)
# df = pd.DataFrame(data, columns=[f'var{i+1}' for i in range(6)])

# print(df)

# chi_sq, p = calculate_bartlett_sphericity(df)
# print(f'Chi-square: {chi_sq}, P-value: {p}')

# kmo_all, kmo_model = calculate_kmo(df)
# print(f'KMO: {kmo_model}')


# fa = FactorAnalyzer(rotation="varimax")
# fa.fit(df)
# eigen_values, _ = fa.get_eigenvalues()

# print(eigen_values)
# plt.plot(range(1, df.shape[1]+1), eigen_values, marker='o')
# plt.title('Scree Plot'); plt.xlabel('Factors'); 
# plt.ylabel('Eigenvalue'); plt.grid(True); plt.show()

# fa = FactorAnalyzer(n_factors=2, rotation="varimax")
# fa.fit(df)

# print("Factor Loadings:\n", fa.loadings_)
# print("Factor Variance:\n", fa.get_factor_variance())
# print("Factor Scores:\n", fa.transform(df))

#Kaggle time

dftrain = pd.read_csv('train.csv',index_col=0)
dftest = pd.read_csv('test.csv',index_col=0)
df_ori = pd.concat([dftrain, dftest], sort=False)

df_ori.info()

df = df_ori.copy()
df = df.iloc[:,6:23]
df = df.dropna()

df.info()

print("Null ",df.isnull().sum())

sns.pairplot(df, diag_kind='kde')
#plt.show()

def grab_column_names(df, cardinality_threshold=20):
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist() 
    
    cardinal_columns = [col for col in categorical_columns if df[col].nunique() > cardinality_threshold]
    
    return numeric_columns, categorical_columns, cardinal_columns 

numeric_columns, categorical_columns, cardinal_columns = grab_column_names(df)     
print("Numeric columns :", numeric_columns,)
print("Catagorical Columns:", categorical_columns )

def plot_histograms(df, columns, rows=2, cols=3):
    """
    Plot histograms for specified columns in a dataframe.

    Args:
    df (pandas.DataFrame): The dataframe containing the data.
    columns (list): List of column names to plot.
    rows (int): Number of rows in the plot grid. Default is 2.
    cols (int): Number of columns in the plot grid. Default is 3.
    """ 
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()  # Flatten axes array for easy indexing

    for i, col in enumerate(columns):
        if i < len(axes):  # Check if we have enough subplots
            sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
            axes[i].set_title(col)
            axes[i].set_xlabel('')  # Remove x-label as it's redundant with title

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show() 

plot_histograms(df, numeric_columns, rows=4, cols=4)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()