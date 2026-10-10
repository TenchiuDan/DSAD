import pandas as pd
import matplotlib.pyplot as plt

# Load the file
df = pd.read_csv("uk-gdp-growth-and-unemployment-rate.csv")

# Quick structural inspection
print(df.head())      # Preview first 5 rows
print(df.info())      # Column data types & non-null counts
print(df.shape)     # Dimensions: (50 rows, 5 columns)




plt.figure(figsize=(8, 5))
plt.scatter(
    df["Unemployment rate (%)"],
    df["GDP Growth (%)"],
    color="#2b5c8f",
    alpha=0.75,
    edgecolors="black",
    linewidths=0.5,
)

# Rename columns to cleaner identifiers
df = df.rename(columns={
    'Unemployment rate (%)': 'unemployment_rate',
    'GDP Growth (%)': 'gdp_growth',
    'Entity': 'entity',
    'Code': 'country_code',
    'Year': 'year'
})


df_clean = df.drop(columns=['entity', 'country_code'])

df_clean.isna().sum()  # Confirms 0 null values across all columns


recessions = df_clean[df_clean['gdp_growth'] < 0]
# Returns years like 1974, 1975, 1980, 2008, 2009, 2020

# Periods with unemployment > 8% AND positive GDP growth
recovery_years = df_clean[(df_clean['unemployment_rate'] > 8.0) & (df_clean['gdp_growth'] > 0)]


df_clean[['unemployment_rate', 'gdp_growth']].describe()

plt.title("UK: Unemployment Rate vs. GDP Growth (1971–2020)", fontsize=13)
plt.xlabel("Unemployment Rate (%)", fontsize=11)
plt.ylabel("GDP Growth (%)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)


plt.tight_layout()
plt.show()