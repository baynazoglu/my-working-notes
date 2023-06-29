#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("DERSLER/RECOMMENDATION SYSTEMS/DATASETS/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape
df.columns

#tfid vectorizerda commentler eng. oldugu icin stop_wordsleri  belirttik. ing deki edatları vs (the,of,in vs silecek)
tfidf = TfidfVectorizer(stop_words="english")

# df[df['overview'].isnull()] #bos olanlara baktık.
df['overview'] = df['overview'].fillna('') #bosları boslukla doldurduk.

tfidf_matrix = tfidf.fit_transform(df['overview']) #tfidf matrisi yaptık.

tfidf_matrix.shape #commentler satır, commentlerin tek tek değerleri sutun oldu.

df['title'].shape
import numpy as np

columns_drop= ['id','imdb_id', 'original_language', 'original_title', 'overview','popularity', 'title','vote_average', 'vote_count']
df = df[columns_drop]
df.head()
#burada bunları yaptık cunku dfin kendisi cok buyuk yer kaplıyor. biz verisetini kücülttük.

tfidf.get_feature_names()
tfidf_matrix.astype(np.float32).toarray()
# tfidf_matrix.toarray() hata veriyordu yukarıdakiyle boyut kücültmüş olduk.


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)
cosine_sim = cosine_similarity(tfidf_matrix.astype(np.float32),
                               tfidf_matrix.astype(np.float32))

cosine_sim.shape
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
