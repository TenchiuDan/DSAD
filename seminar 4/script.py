import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("test")


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(1,1,1)
ax.set_title("Weekly Evolution")
ax.set_xlabel("Day")
ax.set_ylabel("Value")
days=["M","Tu","W","Th","F","Sa","Su"]
values=[12,15,11,18,22,20,9]
ax.plot(days,values,marker="o",color="blue")
ax.axhline(np.mean(values),color="red",linestyle="--", label="Mean")
plt.show()

#2 scatter plot with labels
t = pd.DataFrame({
    "comp1":[-2.1,0.5,1.8,-0.3],
    "comp2":[0.4,-1.2,0.9,1.5],
},index=["RO","HU","DE","FR"])

plt.figure(figsize=(9,7))
plt.scatter(t["comp1"],t["comp2"])

print(list(zip(t.index, t["comp1"], t["comp2"])))

# for label, x, y in zip(t.index, t["comp1"], t["comp2"]):
#     plt.text(x, y, label, fontsize=10, ha='right', va='bottom')


plt.xlabel("comp1")
plt.ylabel("comp2")
plt.title("Scatter plot")
plt.grid(True)
plt.show()



#3 HeatMap
import seaborn as sns

#correlation 
r = np.array([
    [1.0,72,-0.15],
    [0.72,1.0,0.31],
    [-0.15,0.31,1.0]
])

labels = ["Var1", "Var2", "Var3"]
df_r = pd.DataFrame(r, index=labels, columns=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(df_r, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()



#Principal component analysis(PCA)

import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("vikrishnan/iris-dataset")

# print("Path to dataset files:", path)

T = pd.read_csv("iris.data.csv", header=None)

T.columns=["SepalLemgthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]

X=T.iloc[:,:4]

X=T[["SepalLemgthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

X_scaled=StandardScaler().fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

result=pd.DataFrame(X_pca,columns=["PCA1","PCA2"])
result["Species"]=T["Species"]

#see how much variance is perserved
print("Explained variance: ", pca.explained_variance_ratio_)
print("Total variance",pca.explained_variance_.sum())

for species, group in result.groubBy("Species"):
    plt.scatter(group["PCA1"], group["PCA2"], label=species)


plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA1 vs PCA2")
plt.legend()

plt.show()
