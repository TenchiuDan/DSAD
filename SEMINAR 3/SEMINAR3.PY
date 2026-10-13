import pandas as pd

df_cerinta1=pd.read_csv("IndustriaAlimentara.csv")

activities = df_cerinta1.columns[2:]
#we take the third and rest of the columns cause only that data is interested to us

print(activities)

filtered_df= df_cerinta1[df_cerinta1[activities].sum(axis=1)>0]
#the axis chooses how it adds up, that is by horiontal or vertical


filtered_df.to_csv("Cerinta1.csv", index = False)

#NUMBER @@@@2

#still use same activities list

total_employees= df_cerinta1[activities].sum(axis=1)
percentages= df_cerinta1[activities].div(total_employees,axis=0).mul(100)


percentages=percentages.fillna(0)

df_cerinta2=df_cerinta1.copy()

df_cerinta2[activities]=percentages
df_cerinta2.to_csv("Cerinta2.csv", index=False)

#NUMBER 3

# we got to merge now 
df_populatie=pd.read_csv("PopulatieLocalitati.csv")

print("\nIndustria: \n",df_cerinta1.columns)

print("\n Population tabke: \n",df_populatie.columns)

#MERGING
df_merged = pd.merge(df_cerinta1,df_populatie, on="Siruta",how="inner")

#check if it looks good
df_merged.to_csv("MErgedCerinta3.csv")

#check if we did good
print(df_merged.head())
#yes we did


filtered_merge=df_merged.groupby("Judet")[list(activities)+["Populatie"]].sum()


filtered_merge["TotalEmployees"]=filtered_merge[activities].sum(axis=1)

filtered_merge["Pondere"]=filtered_merge["TotalEmployees"]/filtered_merge["Populatie"]*100

filtered_merge=filtered_merge.sort_values(by="Pondere",ascending=False)

filtered_merge.to_csv("Cerinta3Final.csv")

#NUMERO 4

print(df_merged,end="\n\n")

cerinta4_df=df_merged.groupby("Judet")[activities].sum()

print(cerinta4_df)

cerinta4_df["activitateDominant"]=cerinta4_df[activities].idxmax(axis=1)

print(cerinta4_df[["activitateDominant"]])

cerinta4_df[["activitateDominant"]].to_csv("Cerinta4.csv")
