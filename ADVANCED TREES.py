################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
#mesela, max_samples,max_depth,n_estimators(fit edilecek agac sayısı) bunlar onemli. criterion: gini bu da değişilebilir.

#once oto degerlere göre bi accuracy f1 roc sonuclarını alalım...

cv_results = cross_validate(rf_model,X,y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.753
cv_results['test_f1'].mean()
#0.619
cv_results['test_roc_auc'].mean()
#0.823

rf_params = {"max_depth": [5,8,None],
             "max_features": [3,5,7,"auto"],
             "min_samples_split":[2,5,8,15,20],
             "n_estimators": [100,200,500]}
#eger rf_paramslı best hyperparametreli degerde hata oranı daha kötü cıkarsa neden?
#1. rf paramsta hyperparametrelere oto atanan degerleri de yazzmak lazım cunku onlar da en iyi degerler olabilirr..
#ya da sectigimiz random degerle alakalıdır. rand state 17 dedik mesela onla alakalıdır. 41 yap mesela.
rf_best_grid = GridSearchCV(rf_model,rf_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
#900 tane fiti var ve her agacta birsürü dal var. uzun surmesi normal..

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state= 17).fit(X,y)

cv_results = cross_validate(rf_final,X,y, cv=10,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
#0.7668489
cv_results['test_f1'].mean()
#0.6447777811143756
cv_results['test_roc_auc'].mean()
#0.8271054131054132

#degerler güzellesti..

#daha neler yapabiliriz?
#her ml modelinde grafikleştirme ya da feature importance yapabiliriz.
# bu modelde de yapacagız ama ilerleyenlerde gostermeyecegiz.

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

plot_importance(rf_final,X)
#glucose en elzem.

#roc curve icin...

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


val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")

################################################
# GBM
################################################

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
#learningrate var gradient descent temelli oldugu icin.
#artık tahminin basındaki kat sayı.
#n-estimators(optimizasyon sayısı aslında)
#sub sample,verbose vsvs.

#hyperparametre optimizasyonu oncesi degerler:

cv_results = cross_validate(gbm_model,X,y,cv=5,scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7591715474068416
cv_results['test_f1'].mean()
# 0.634
cv_results['test_roc_auc'].mean()
# 0.82548

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
#mesela subsample ne? verideki gözlemin kaçını kullanayım demek.
# ileride ne oldugunu unuttun mu? hemen gbmnin acıklamsana git ve oku.

gbm_best_grid = GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
gbm_best_grid.best_params_


gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#7800186741363212
cv_results['test_f1'].mean()
#668605747317776
cv_results['test_roc_auc'].mean()
#8257784765897973

################################################
# XGBoost
################################################
#GBM'in hız ve tahmin açısından performansını arttırmak için optimize edilmiş, ölçeklenebilir
#ve farklı platformlarda entegre edilebilecek bir versiyondur. bir bilgisayaracı inşaa etmiştir. 2017ye kadar ML kralıydı ama sonra light gbm geldi....

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.75265
cv_results['test_f1'].mean()
# 0.631
cv_results['test_roc_auc'].mean()
# 0.7987

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}
#colsample dedigi subsample aslında..

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#75
cv_results['test_f1'].mean()
#60
cv_results['test_roc_auc'].mean()
#82

################################################
# LightGBM
################################################
#Xgboost dallara ayırılırken geniiş kapsamlı bir arama yaparken,
#lightgbm, derinlemesine ilk arama yapmaktadır..
#bu sebeple en hızlı olan modeldir.  microsoft 2017.


lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.74744928274340
cv_results['test_f1'].mean()
#624110522144179
cv_results['test_roc_auc'].mean()
#7990293501048218
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
#lgbm icin en kritik hyperparam n_estimatordur.
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.7643578643578645

cv_results['test_f1'].mean()
#6372062920577772

cv_results['test_roc_auc'].mean()
#8147491264849755

# Hiperparametre optimizasyonu sadece n_estimators için. her lgbmde bunu yapmak lazım. n_est cok kritik.

lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7643833290892115
cv_results['test_f1'].mean()
#6193071162618689
cv_results['test_roc_auc'].mean()
#8227931516422082
################################################
# CatBoost
################################################
#yandex 2017

#categoring boosting. kat. degiskenlerle hızlıca ve otomatik olarak mücadele edebilen,hızlı ve başarılı diğer gbm türevi.

#biz eda'da napıyorduk? eger cat deg. one hot encoding ya da label encoding vs. bazen siliyorduk.

catboost_model = CatBoostClassifier(random_state=17, verbose=False)
#verbose false yapıyoruz cunku true yaparsak cıktısı cok cırkın :).
#süresi coook uzun olabiliyor catboostun...

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

#hepsinde feature importance aynı sıralamada cıktı. ama lightgbmde glucose ilk sırada degil bmi var.
#cok columnlu yani featurelı verilerde buna bakmak çok önemli cunku her columnu eklemeye gerek yok.
#grafiğe göre en elzemlerle yola devam edilebilir. biz kendimiz de column uretecegiz. bakarız mlye bi katkısı var mı yok mu diye?



#################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################

#biz gridsearchle yapıyorduk hiperparamtre optimizasyonu... bu da alternatif...

#mesela aralıklarla alakalı hiçbir bilgimiz yoksa.
# once randomsearchle aralıklar girilip random arama yapılabilir. cıkan degerlerin alt üst degerler ekleyip bi gridsearche sokabiliriz...


rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}
#mesela max depth icin 5le 50 arası 10 tane random sayı verdik. estimators icin 200den 1500e 10ar 10ar artarak cıktık.
#birsürü deger var ama n_iter ile kaç tane sececegini ve onlar arasından best degeri getirecegini yazacagız.


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
#cv_results['test_accuracy'].mean()
# 0.7682964094728801
#cv_results['test_f1'].mean()
# 0.6297764182815141
#['test_roc_auc'].mean()
#0.8360859538784066

################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################
#biz hyperparamları zaten sectik.bunu niye yapıyoruz? grafikle görüyoruz hangi degerler dogru olanlar....

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


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]

