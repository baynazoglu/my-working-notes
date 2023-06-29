################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################
#bazı kaynaklar prediction yerine scoring de der.
import joblib
import pandas as pd

df = pd.read_csv("datasets/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
#boyut hatası verdi neden? biz cünkü verinin altından girip üstünden cıktık. yeni degiskenler ekledik vs. farklı bi boyuta  evrildi
# ama su an predict etmek istedigimiz veri ham halindeki gibi. onu da data prep asamasına sokmamız lazımdır.


from diabetes_pipeline import diabetes_data_prep

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
