################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

df= pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/USArrests.csv",index_col=0)
# index 0 eyalet adlarıydı boyle yazınca daha proper oldu

df.head()
df.isnull().sum()
df.info()

#merkeze olan uzaklıklara bakacagız k-meanste o sebeple scale yapalım...


sc= MinMaxScaler((0,1))
dff = sc.fit_transform(df)
dff[0:5]

kmeans = KMeans(n_clusters=4).fit(dff)
#n_clusters olusturulacak kume sayısıydı.burada 4 verdik.

kmeans.get_params()
#max iter 300 mesela..
kmeans.n_clusters
kmeans.cluster_centers_ #merkeze olan uzaklıkları...
kmeans.labels_ #hangi kümelerdeler...
kmeans.inertia_#inertia demek sse sonucu. sum of square error. yani toplam hata karaleri..

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(dff)
    ssd.append(kmeans.inertia_)

    #boylece 1den 30a kadar her sse yi aldık. bakalım hangisi bizim optimumuzz.

plt.plot(K,ssd,"bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show(block=True)

#gorduk ki k arttıkca hata karaler dusuyor. ama 30 gözlem icin 30 kume de sacma.
#burada bizim kendi yorumumuz onemli. verisetine göre kendimiz karar vermeliyhiz...
#ama yine de matematiğin yardımıyla bir hesaplama yöntemi de var...


kmeans= KMeans()
elbow = KElbowVisualizer(kmeans,k=(2,20))
elbow.fit(dff)
elbow.show()

elbow.elbow_value_
#burada 7dedi. 7 kümeli olmalı...


################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(dff)
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
dff[0:5]


cluster_kmeans = kmeans.labels_
#kümeleri cluster_kmeans diye bir degiskene atadık..

df["clusters"] = cluster_kmeans

df.head()

#İLK KÜME 0 DEGİL DE 1 OLSUN DİYORSAK DA.

df["clusters"] = df["clusters"] + 1
df.head()
df[df["clusters"]==1]
#1 olanları gorduk. 4 eyalet varmış...

df.groupby("clusters").agg(["count","mean","median"])

#eger bunu bir excele cevirmek istersek...
df.to_csv("clusters.csv")

################################
# Hierarchical Clustering
################################

#birleştirici ve ayrıştırıcı kümeleme vardı...

df= pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/USArrests.csv",index_col=0)

sc = MinMaxScaler((0,1))
dff = sc.fit_transform(df)

hc_average = linkage(df,"average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show(block=True)

#dendogram yaptık. ve her gözleme yer verdik...
#gözlem sayısını dusursek mesela..

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show(block=True)

################################
# Kume Sayısını Belirlemek
################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

#line çektik 0.5ten ve gördük ki 5 tane küme oldu.
#0.6dan cektik 4e indi küme sayısı. karar biizm.

################################
# Final Modeli Oluşturmak
################################


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
df["kmeans_cluster_no"] = clusters_kmeans

################################
# Principal Component Analysis
################################
#bilgi kaybını göze alarak gözlem düşürme işi...


df= pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/hitters.csv")
df.head()
#normalde bu bir maaş hesaplama veriseti. ve birsürü cat degisken null value var.
#biz burada bagımlı degiskene ya da cat degiskenlere bkamıyoruz. amac int degiskenlerden gözlemleri azaltmak. hata payını göze alarak.

num_cols = [col for col in df.columns if df[col].dtypes !="O" and "Salary" not in col]
num_cols

df = df[num_cols]
df.dropna(inplace=True)
df.shape

dff = StandardScaler().fit_transform(df)
dff[0:5]

pca = PCA()
pca_fit = pca.fit_transform(dff)

pca.explained_variance_ratio_
#variance ratio bizim modelimizin ne kadar iyi olup olmadıgını veren değerdir
# variance ratio : hata payı yani ilk gözlem tek başına verisetini %45 yansıtıyor. 2. ise %26 vs. bunların toplamına bakalım...

np.cumsum(pca.explained_variance_ratio_)
#yani ilk 3 gözlem toplam veri setinin %82sini yansıtıyor. gayet iyii...
################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(dff)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show(block=True)

################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(dff)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)
################################
# BONUS: Principal Component Regression
################################

df = pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/hitters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)
final_df.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
y.mean()


cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))


################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")



















