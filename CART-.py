################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model


# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. Exploratory Data Analysis
################################################
#diabetes veri setiyle calısalım ve daha onceden yaptıgımız eda ve feature eng.leri alalım.
df = pd.read_csv("DERSLER/Feature Engineering/case_study_diabetes/diabetes.csv")
df.head()

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """

       Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
       Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

       Parameters
       ------
           dataframe: dataframe
                   Değişken isimleri alınmak istenilen dataframe
           cat_th: int, optional
                   numerik fakat kategorik olan değişkenler için sınıf eşik değeri
           car_th: int, optinal
                   kategorik fakat kardinal değişkenler için sınıf eşik değeri

       Returns
       ------
           cat_cols: list
                   Kategorik değişken listesi
           num_cols: list
                   Numerik değişken listesi
           cat_but_car: list
                   Kategorik görünümlü kardinal değişken listesi

       Examples
       ------
           import seaborn as sns
           df = sns.load_dataset("iris")
           print(grab_col_names(df))


       Notes
       ------
           cat_cols + num_cols + cat_but_car = toplam değişken sayısı
           num_but_cat cat_cols'un içerisinde.
           Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

       """
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" ]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#bakmak istiyorsak diye..
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


        for col in age_cols:
            num_summary(df,col,True)
num_summary(df,num_cols,plot=True)
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
cat_summary(df,"Outcome")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

def outlier_thresholds(dataframe, colname, q1 =0.25, q3= 0.75):
    quartile1 = dataframe[colname].quantile(q1)
    quartile3 = dataframe[colname].quantile(q3)
    interquartile = quartile3 - quartile1
    up_limit = quartile3 + interquartile * 1.5
    low_limit = quartile1 - interquartile * 1.5
    return  low_limit, up_limit
def check_outlier(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    if dataframe[(dataframe[colname] < low ) | (dataframe[colname] > up)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    df.loc[(df[colname] > up), colname] = up
    df.loc[(df[colname] < low), colname] = low
for col in num_cols:
    print(col, check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)

def missing_values_table(dataframe,na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df,True)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

missing_values_table(df)

def missing_vs_target(dataframe,target,na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(),1,0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
missing_vs_target(df,"Outcome",zero_columns)

for col in zero_columns:
    df.loc[df[col].isnull(),col] = df[col].median()

df.loc[(df["Age"] < 45), "NEW_AGE_CAT"] ="mature"
df.loc[(df["Age"] > 45), "NEW_AGE_CAT"] ="senior"

#bmi chartına göre 18.5 is underweight, 18.5 to 24.9 is normal, 24.9 to 29.9 is Overweight, and over 30 is obese
df.head()
df.loc[(df["BMI"]> 0) & (df["BMI"] < 18.5), "NEW_BMI_CAT"] ="underweight"
df.loc[(df["BMI"]>= 18.5) & (df["BMI"] < 24.9), "NEW_BMI_CAT"] ="normal"
df.loc[(df["BMI"]>= 24.9) & (df["BMI"] <= 29.99), "NEW_BMI_CAT"] ="overweight"
df.loc[(df["BMI"]>= 30), "NEW_BMI_CAT"] ="obese"

#glikoz değeri için de bir ayırım var.
df["Glucose"].describe().T
df.loc[(df["Glucose"]> 0) & (df["Glucose"] < 140), "NEW_GLUCOSE_CAT"] ="normal"
df.loc[(df["Glucose"]>= 140) & (df["Glucose"] < 200), "NEW_GLUCOSE_CAT"] ="prediabetic"
df.loc[(df["Glucose"]>= 200), "NEW_GLUCOSE_CAT"] ="diabetic"

df["NEW_GLUCOSE_CAT"].nunique()

#diabetik olan yokmuş. maks deger 199du cunku

#bu yaptıklarımızı yaşa göre kıralım.

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "obesesenior"


df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "highsenior"

#encoding...
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df,col)

##### one hot encoding...

def one_hot_encoder(dataframe,categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]


df = one_hot_encoder(df, ohe_cols, drop_first=True)


#Numerik değişkenler için standartlaştırma

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

#df[num_cols] = scaler.inverse_transform(df[num_cols])

################################################
# 3. Modeling using CART
################################################
df = pd.read_csv("DERSLER/Feature Engineering/case_study_diabetes/diabetes.csv")
y= df["Outcome"]

X= df.drop("Outcome",axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X,y)
#Confusion Matrix için y_pred:
y_pred = cart_model.predict(X)
#AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:,1]

#Confusion Matrix:
print(classification_report(y,y_pred))
#1 çıktı. %100 olmus olamaz degil mi?

#AUC
roc_auc_score(y, y_prob)
#Bu da 1 çıktı...

#hold-out ve cv deneyelim bakalım onlar da 1 mi?
#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


#####################
# CV ile Başarı Değerlendirme
#####################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347

#model validation yaptık... gördük ki cross validationda hassas değerler geldi ama diğerlerinde overlearning vardı
#peki değerleri gördük. 0.57, 0.7 vs bu değerleri nasıl yükseltebiliriz?
# veri ekleyebilirdik,hyperparametre optimizasyonu yapabilirdik vsvs. burada hiper yapacagız.


################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

cart_model.get_params()
#min samples 2.  max depth none. bunlar kendinden atanan degerler.
# bunlarla oynayıp optimize edip en iyi sonucu alabiliriz.. nasıldı?

cart_params = {"max_depth":range(1,11),
               "min_samples_split": range(2,20)}

cart_best_grid = GridSearchCV(cart_model,cart_params,cv=5,n_jobs=-1,verbose=1).fit(X,y)
#burada genel x ve y ye fitledik. x_train y_train falan da olabilirdi ama
#zaten veri azdı. en genele fitlemek daha mantıklıydı.

cart_best_grid.best_params_
#max_depth 5 min samp:2 cıktı.
cart_best_grid.best_score_
#0.74 cıktı best score. kendi score olarak accuracy alıyor onu da degisebiliriz...


random = X.sample(1,random_state=45)

cart_best_grid.predict(random)
#1 diye tahmin etti..
#burada tekrar işlem yapmadık cunku cart_best_grid zaten best değerlerle bi öğrenim yapmıştı
#direkt onu çektik.
#ama best hyperparametersları ogrendik. bi final model yazabiliriz...

################################################
# 5. Final Model
################################################

#2 türlü yapabiliriz.

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)
#bunu yaptık cunku zaten cart modeli tanıtmıştık.** yaparak o degeleri al dedik.

#ya da

cart_final  = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X,y)

cart_final.get_params()

cv_results = cross_validate(cart_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
#0.74
cv_results['test_f1'].mean()
#0.61
cv_results['test_roc_auc'].mean()
#0.79

#gördügümüz gibi final modelde modeli iyileştirdik. 0.70 olan accuracyi 0.75e cıkardık.....
#peki cart modellerinde en üstteki dal en degerli, önemli olandı. o hangisi??


################################################
# 6. Feature Importance
################################################x

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(cart_final,X, num=5)
#gördük ki gloucose en onemli degisken...


################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################


train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring="roc_auc",
                                           cv=10)
#curvede bir noktadan sonra train overfittinge baslıyordu. o noktayı grafikle gorecegiz simdi.
mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)


plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show(block=True)


#yine bir func yazıldı...

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

#burada sadece tek bir param için sonuç alıyoruz.
#biz 2 hyperparam için bir optimizasyonda 5 ve 3 sonuclarını almıstık. max depth ve diğeri.
#ama tek tek bakınca 5 yerine 3 geliyor. 2sinide bir dfe koyalım. sonra for donguyle ona bakalım.

val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])



################################################
# 8. Visualizing the Decision Tree
################################################
# conda install graphviz
# import graphviz
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)
tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

cart_final.get_params()

################################################
# 9. Extracting Decision Rules
################################################
#dallanmaları consolda gosterecegiz.

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

################################################
# 10. Extracting Python Codes of Decision Rules
################################################

import sklearn
sklearn.__version__

# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))

#boylelıkle python için sqlite icin ve excel icin calıstıgımız işi cevirdik.
#yani sqliteda cıkan sonucu sqlite a koyarsak orada bu calısmamızı görecegiz.

################################################
# 11. Prediction using Python Codes
################################################

#bu işe tekrar bakmak istedigimizde ya da,
# birine gönderdigimizde ya da yeni biri eklendiginde
#her defasında tek tek kod çalıştırmak saçma. buna bi func yazalım..

def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns

x = [12, 13, 20, 23, 4, 55, 12, 7]

predict_with_rules(x)

x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)

#artık istedigimiz degerleri girerek tahmini hemen bu funcla yapabiliriz...


################################################
# 12. Saving and Loading Model
################################################
#bu modeli kaydedip birityle paylaşmak istersek eğer...

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]

cart_model_from_disc.predict(pd.DataFrame(x).T)

#burada transpose aldık cunku modele girecegimiz degerin dataframe olması lazım. biz liste girdik onu pd yapıp transpoze yaptık.


