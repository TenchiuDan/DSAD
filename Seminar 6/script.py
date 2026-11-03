print("test")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from factor_analyzer import (FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo)

rng = np.random.default_rng(42)
n=300
latent_factors=rng.normal(size=(n, 3))

questions={
    "Price_1":(0,0.90),
    "Price_2":(0,0.85),
    "Price_3":(0,0.80),
    "Quality_1":(1,0.90),
    "Quality_2":(1,0.85),
    "Quality_3":(1,0.80),
    "Service_1":(2,0.90),
    "Service_2":(2,0.85),
    "Service_3":(2,0.80)
}

data={}

for name, (factor_index,loading) in questions.items():
    data[name]=(
        loading* latent_factors[:,factor_index]+np.sqrt(1-loading**2)*rng.normal(size=n)
                )
df=pd.DataFrame(data)
print("first 5 rows of the dataset:")
print(df.head())

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", center=0)
plt.title("Correlation Heatmap of Simulated Survey Data")
plt.show()

# Bartlett's test of sphericity

chi_square,p_value = calculate_bartlett_sphericity(df)
print("bartlett's test of sphericity:")
print("Chi - sqaure test",round(chi_square,2))
print("p-value",round(p_value,2))

print(chi_square )
if p_value < 0.05:
    print("Reject the null hypothesis: The correlation matrix is not" \
    " an identity matrix. Factor analysis may be appropriate.")
else: 
    print("Fail to reject the null hypothesis: The correlation matrix "
    "is an identity matrix. Factor analysis may not be appropriate.")

    #KMO TEST

kmo_items, kmo_total = calculate_kmo(df)

print("KMO TEST")
print("Overall KMO:", round(kmo_total,3))

kmo_table=pd.Series(kmo_items, index=df.columns, name="KMO")

print(kmo_table.round(3))

#eigen values and scree plot
fa_initial = FactorAnalyzer(
    n_factors=df.shape[1], 
    rotation=None, 
    method="principal"
    )

fa_initial.fit(df)
eigen_values, _ = fa_initial.get_eigenvalues()

print("Eigenvalues:")
print(np.round(eigen_values, 3))
plt.figure(figsize=(10,6))
plt.plot(range(1,len(eigen_values)+1), eigen_values)
plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.axhline(y=1, color='r', linestyle='--')
plt.show()