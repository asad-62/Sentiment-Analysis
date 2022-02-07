# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 21:11:08 2021

@author: Asad
"""
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import  RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
######
df_review = pd.read_csv('IMDB Dataset.csv')
df_positive = df_review[df_review['sentiment']=='positive'][:9000]
df_negative = df_review[df_review['sentiment']=='negative'][:1000]

df_review_imb = pd.concat([df_positive, df_negative])
rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']],
                                                           df_review_imb['sentiment'])
train,test=train_test_split(df_review_bal,test_size=0.33,random_state=42)
train_x=train['review']
train_y=train['sentiment']
test_x=test['review']
test_y=test['sentiment']
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

### with svm
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)



log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

svc.score(test_x_vector, test_y)

print("SVM ACCURACY",svc.score(test_x_vector, test_y))

log_reg.score(test_x_vector, test_y)

print(classification_report(test_y, 
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative']))

conf_mat = confusion_matrix(test_y, 
                            svc.predict(test_x_vector), 
                            labels=['positive', 'negative'])


