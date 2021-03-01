import numpy as np
import pandas as pd
from collections import defaultdict
import re
import string
from bs4 import BeautifulSoup
import sys
import os
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import itertools
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import nltk
import collections
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from empath import Empath
from keras.preprocessing.text import Tokenizer
from nltk import tokenize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle






def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = "".join(c for c in string if c not in ('!',':','#','@','$','"','%','&','(','-',')','*','+',',','/',';','<','=','>','[','^',']','_','`','{','|','}','~'))
#    ,'?','.'                                                
    return string.strip().lower() 




#Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   




#********************************************************************
#********************************************
#***********************    
df_fake_base_complete = pd.read_csv(r"E:\FakeNews\FakeNews-code\keras\HAN\Dr.Zbihzadeh\data\fake_base_complete.csv", encoding='latin-1')
df_fake_base_complete.columns = ['Tweet', 'Label']
df_fake_base_complete['Label'].value_counts()


#+++++++++++++++++++++++++++++++++++
df1_test = pd.read_csv(r"D:\Doctora99\Doctora-term1-9908\ML.DrManthouri\HW\HW&Project\Kaggle\111\test.csv", encoding='latin-1')
df2_test = pd.read_csv(r"D:\Doctora99\Doctora-term1-9908\ML.DrManthouri\HW\HW&Project\Kaggle\111\test_labels.csv", encoding='latin-1')

df1_test.columns = ['ID', 'Tweet']
df1_test = df1_test[['Tweet']]
df2_test = df2_test[['toxic']]
df2_test.columns = ['Label']

test_labeled = pd.concat([df1_test, df2_test], axis=1)
test_labeled["Label"]= test_labeled["Label"].replace(0, 1.0)

df_test = test_labeled


df_test = df_test.replace(r'^\s*$', np.NaN, regex=True)
df_test = df_test.dropna(axis = 0, how ='any')
#df_test = df_test.drop_duplicates()
df_test = df_test.reset_index(drop=True)
df_test['Label'].value_counts()
df_test['Tweet'].describe()

plt.figure()
df_test['Label'].value_counts().plot(kind = 'bar')
df_test["Label"]= df_test["Label"].replace(0, -1.0)

#++++++++++++++++++++++++

df1_train = pd.read_csv(r"D:\Doctora99\Doctora-term1-9908\ML.DrManthouri\HW\HW&Project\Kaggle\111\train.csv", encoding='latin-1')
df1_train = df1_train[['comment_text','toxic']]
df1_train.columns = ['Tweet', 'Label']
df1_train["Label"]= df1_train["Label"].replace(0, -1.0)

df1_train = df1_train.replace(r'^\s*$', np.NaN, regex=True)
df1_train = df1_train.dropna(axis = 0, how ='any')
#df1_train = df1_train.drop_duplicates()
df1_train = df1_train.reset_index(drop=True)
df1_train['Label'].value_counts()
df1_train['Tweet'].describe()

plt.figure()
df1_train['Label'].value_counts().plot(kind = 'bar')
df1_train["Label"]= df1_train["Label"].replace(0, -1.0)


df_train, df_validation = train_test_split(df1_train, test_size=0.2, random_state=42, shuffle=True)


df_validation = df_validation.replace(r'^\s*$', np.NaN, regex=True)
df_validation = df_validation.dropna(axis = 0, how ='any')
#df_validation = df_validation.drop_duplicates()
df_validation = df_validation.reset_index(drop=True)
df_validation['Label'].value_counts()
df_validation['Tweet'].describe()

plt.figure()
df_validation['Label'].value_counts().plot(kind = 'bar')
df_validation["Label"]= df_validation["Label"].replace(0, -1.0)





x_train = df_train['Tweet']
y_train = df_train['Label']
x_val = df_validation['Tweet']
y_val = df_validation['Label']

   
y_train.value_counts()    
y_val.value_counts()

x_train.describe()
#x_train = x_train.reset_index(drop=True)

y_train.describe()
#y_train = y_train.reset_index(drop=True)

x_val.describe()
#x_val = x_val.reset_index(drop=True)

y_val.describe()
#y_val = y_val.reset_index(drop=True)

#-------------------------------------------------

x_test = df_test['Tweet']
y_test = df_test['Label']
    
x_test.value_counts()    
x_test.describe()
#x_test = x_test.reset_index(drop=True)


y_test.value_counts()
y_test.describe()
#y_test = y_test.reset_index(drop=True)


#-----------------------------------------------------------------------
#--------------------------------------------
#-------------------------
# Classic CLF Model
    

#Tf-idf Bigrams
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1)) 
tfidf1_train = tfidf_vectorizer.fit_transform(x_train.astype('str'))
tfidf_test = tfidf_vectorizer.transform(x_test.astype('str'))


#Top 10 tfidf bigrams 
tfidf_vectorizer.get_feature_names()[-10:]
tfidf_vocabulary = tfidf_vectorizer.vocabulary_
#tfidftrain = tfidf1_train.todense()
#tfidftest = tfidf_test.todense()

#1111111111111111111111111111111111111111111111
clf = RandomForestClassifier()
#clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(kernel='linear')
#clf = GradientBoostingClassifier()
#clf = GaussianNB()
#clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_RandomForest', 'REAL_Train_RandomForest'])



#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_RandomForest', 'REAL_Test_RandomForest'])

print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for RandomForestClassifier_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_RandomForestClassifier_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_RandomForestClassifier: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for RandomForestClassifier_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_RandomForestClassifier_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_RandomForestClassifier: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
      
#22222222222222222222222222222222
clf = MLPClassifier(hidden_layer_sizes=(20),max_iter=100)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(kernel='linear')
#clf = RandomForestClassifier()
#clf = GradientBoostingClassifier()
#clf = GaussianNB()
#clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_MLPClassifier', 'REAL_Train_MLPClassifier'])



#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_MLPClassifier', 'REAL_Test_MLPClassifier'])

print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for MLPClassifier_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_MLPClassifier_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_MLPClassifier: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for MLPClassifier_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_MLPClassifier_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_MLPClassifier: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
#3333333333333333333333333333333333333
#clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(kernel='linear')
#clf = RandomForestClassifier()
#clf = GradientBoostingClassifier()
#clf = GaussianNB()
#clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_DecisionTreeF', 'REAL_Train_DecisionTree'])



#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_DecisionTree', 'REAL_Test_DecisionTree'])

print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for DecisionTree_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_DecisionTree_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_DecisionTree: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for DecisionTree_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_DecisionTree_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_DecisionTree: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
#4444444444444444444444444444444444444444
#clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#clf = DecisionTreeClassifier(random_state=0)
clf = SVC(kernel='linear')
#clf = RandomForestClassifier()
#clf = GradientBoostingClassifier()
#clf = GaussianNB()
#clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_SVM', 'REAL_Train_SVM'])


#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_SVM', 'REAL_Test_SVM'])
print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for SVM_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_SVM_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_SVM: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for SVM_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_SVM_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_SVM: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
#555555555555555555555555555555555555555555
#clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(kernel='linear')
#clf = RandomForestClassifier()
clf = GradientBoostingClassifier()
#clf = GaussianNB()
#clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_GradientBoostin', 'REAL_Train_GradientBoostin'])



#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_GradientBoostin', 'REAL_Test_GradientBoostinF'])

print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for GradientBoostin_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_GradientBoostin_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_GradientBoostin: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for GradientBoostin_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_GradientBoostin_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_GradientBoostin: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
#66666666666666666666666666666666666666666
#clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(kernel='linear')
#clf = RandomForestClassifier()
#clf = GradientBoostingClassifier()
clf = GaussianNB()
#clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_GaussianNB', 'REAL_Train_GaussianNB'])



#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_GaussianNB', 'REAL_Test_GaussianNB'])

print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for GaussianNB_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_GaussianNB_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_GaussianNB: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for GaussianNB_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_GaussianNB_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_GaussianNB: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
#7777777777777777777777777777777777777
#clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(kernel='linear')
#clf = RandomForestClassifier()
#clf = GradientBoostingClassifier()
#clf = GaussianNB()
clf = MultinomialNB()
clf.fit(tfidf1_train, y_train)
pred_train_clf = clf.predict(tfidf1_train)
pred_test_clf = clf.predict(tfidf_test)


#confusion matrix train_clf
#classification_report train_clf
score_train_clf = metrics.accuracy_score(y_train, pred_train_clf)
kappa_train_clf = metrics.cohen_kappa_score(y_train, pred_train_clf)
matrix_train_clf = metrics.confusion_matrix(y_train, pred_train_clf)

plt.figure()
cm = metrics.confusion_matrix(y_train, pred_train_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Train_MultinomialNB', 'REAL_Train_MultinomialNB'])



#confusion matrix test_clf
#classification_report test_clf
score_test_clf = metrics.accuracy_score(y_test, pred_test_clf)
kappa_test_clf = metrics.cohen_kappa_score(y_test, pred_test_clf)
matrix_test_clf = metrics.confusion_matrix(y_test, pred_test_clf)

plt.figure()
cm = metrics.confusion_matrix(y_test, pred_test_clf, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_MultinomialNB', 'REAL_Test_MultinomialNB'])


print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for MultinomialNB_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_MultinomialNB_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_MultinomialNB: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report forMultinomialNB_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_MultinomialNB_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_MultinomialNB: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n\n\n\n"
      )
      

#-----------------------------------------------------------------------
#--------------------------------------------
#-------------------------
# Han Model

x_han_train = x_train.tolist()
y_han_train = y_train.tolist()

x_han_train = pd.Series(x_han_train)
y_han_train = pd.Series(y_han_train)




MAX_SENT_LENGTH = 300
MAX_SENTS = 2
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100



x_train_han_sent = []
y_train_han_sent = []

for idx in range(x_han_train.shape[0]):
    sentences = tokenize.sent_tokenize(str(x_han_train[idx]))
    x_train_han_sent.append(sentences)
    y_train_han_sent.append(y_han_train[idx])
    

x_val_han_sent = []
y_val_han_sent = []

for iidx in range(x_val.shape[0]):
    sentences = tokenize.sent_tokenize(x_val[iidx])
    x_val_han_sent.append(sentences)
    y_val_han_sent.append(y_val[iidx])
     
    
    

x_test_han_sent = []
y_test_han_sent = []

for iiidx in range(x_test.shape[0]):
    sentences = tokenize.sent_tokenize(x_test[iiidx])
    x_test_han_sent.append(sentences)
    y_test_han_sent.append(y_test[iiidx])
     
    
    
    
    
Tweets_Tweets_test_han = []
Tweets_trainlist_han = x_train.tolist()
Tweets_vallist_han = x_val.tolist()
Tweets_Tweets_test_han = sorted(Tweets_trainlist_han + Tweets_vallist_han)    
Tweets_Tweets_test_han = pd.Series(Tweets_Tweets_test_han)
 
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(Tweets_Tweets_test_han)

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))




data_train_han = np.zeros((len(x_train_han_sent), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(x_train_han_sent):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data_train_han[i, j, k] = tokenizer.word_index[word]
                    k = k + 1




countsss = 0
data_val_han = np.zeros((len(x_val_han_sent), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(x_val_han_sent):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word in word_index:
                    countsss += 1
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data_val_han[i, j, k] = tokenizer.word_index[word]
                        k = k + 1







countss = 0
data_test_han = np.zeros((len(x_test_han_sent), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(x_test_han_sent):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word in word_index:
                    countss += 1
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data_test_han[i, j, k] = tokenizer.word_index[word]
                        k = k + 1





print(pd.Series(y_train_han_sent).value_counts())



GLOVE_DIR = "E:/FakeNews/FakeNews-code/keras/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))


countssss = 0
embedding_matrix = np.random.random((50000 + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < 50001:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            countssss += 1
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


# building Hierachical Attention network

embedding_layer = Embedding(50000 + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # return mask
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])




y_train_han_sent = y_han_train
y_train_han_sent.value_counts
y_train_han_sent.value_counts()
y_train_han_sent = y_train_han_sent.replace(-1,0)
y_train_han_sent = to_categorical(y_train_han_sent)
print(y_train_han_sent.sum(axis=0))


y_val_han_sent = y_val
y_val_han_sent.value_counts
y_val_han_sent.value_counts()
y_val_han_sent = y_val_han_sent.replace(-1,0)
y_val_han_sent = to_categorical(np.asarray(y_val_han_sent))
print(y_val_han_sent.sum(axis=0))



history = model.fit(data_train_han, y_train_han_sent, validation_data=(data_val_han, y_val_han_sent),
          nb_epoch=3, batch_size=50)





from matplotlib import pyplot

plt.figure()
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='Train_Han')
pyplot.plot(history.history['val_loss'], label='Val_Han')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='Train_Han')
pyplot.plot(history.history['val_acc'], label='Val_Han')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend()

pyplot.show()




y_test_han_sent = y_test
y_test_han_sent.value_counts
y_test_han_sent.value_counts()
y_test_han_sent = y_test_han_sent.replace(-1,0)
y_test_han_sent = to_categorical(np.asarray(y_test_han_sent))
print(y_test_han_sent.sum(axis=0))



# predict probabilities for train set
yhat_probs_train_han = model.predict(data_train_han, verbose=1)
yhat_classes_train_han = yhat_probs_train_han.argmax(axis=-1)

# predict probabilities for test set
yhat_probs_test_han = model.predict(data_test_han, verbose=1)
yhat_classes_test_han = yhat_probs_test_han.argmax(axis=-1)

train_han_loss, train_han_acc = model.evaluate(data_train_han, y_train_han_sent, verbose=1)
test_han_loss, test_han_acc = model.evaluate(data_test_han, y_test_han_sent, verbose=1)




y_train_han = y_han_train
y_train_han = y_train_han.replace(-1,0)

y_test_han = y_test
y_test_han = y_test_han.replace(-1,0)



#confusion matrix train_HAN
#classification_report train_HAN
score_train_han = metrics.accuracy_score(y_train_han, yhat_classes_train_han)
#train_han_loss, train_han_acc = model.evaluate(data_train_han, y_train_han_sent, verbose=1)
kappa_train_han = metrics.cohen_kappa_score(y_train_han, yhat_classes_train_han)
auc_train_han = metrics.roc_auc_score(y_train_han_sent, yhat_probs_train_han)
matrix_train_han = metrics.confusion_matrix(y_train_han, yhat_classes_train_han)

plt.figure()
cm = metrics.confusion_matrix(y_train_han, yhat_classes_train_han, labels=[1, 0])
plot_confusion_matrix(cm, classes=['FAKE_Train_HAN', 'REAL_Train_HAN'])




#confusion matrix test_HAN
#classification_report test_HAN
score_test_han = metrics.accuracy_score(y_test_han, yhat_classes_test_han)
#test_han_loss, test_han_acc = model.evaluate(data_test_han, y_test_han_sent, verbose=1)
kappa_test_han = metrics.cohen_kappa_score(y_test_han, yhat_classes_test_han)
auc_test_han = metrics.roc_auc_score(y_test_han_sent, yhat_probs_test_han)
matrix_test_han = metrics.confusion_matrix(y_test_han, yhat_classes_test_han)

plt.figure()
cm = metrics.confusion_matrix(y_test_han, yhat_classes_test_han, labels=[1, 0])
plot_confusion_matrix(cm, classes=['FAKE_Test_HAN', 'REAL_Test_HAN'])




print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for HAN_train Model\n\n",classification_report(y_train_han, yhat_classes_train_han),
      "\n\n Accuracy_HAN_train:   %0.3f" % score_train_han,
      "\n\n train_HAN_acc: %.3f, train_HAN_loss: %.3f" %(train_han_acc, train_han_loss),
      "\n\n Cohens kappa_train_HAN: %f" % kappa_train_han,
      "\n\n ROC AUC of Train_HAN : %f" % auc_train_han,
      "\n\n",matrix_train_han,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for HAN_test Model\n\n",classification_report(y_test_han, yhat_classes_test_han),
      "\n\n Accuracy_HAN_test:   %0.3f" % score_test_han,
      "\n\n test_HAN_acc: %.3f, test_HAN_loss: %.3f" % (test_han_acc, test_han_loss),
      "\n\n Cohens kappa_test_HAN: %f" % kappa_test_han,
      "\n\n ROC AUC  of Test_HAN : %f" % auc_test_han,
      "\n\n",matrix_test_han,
      "\n\n\n\n\n\n\n\n"
      )
      


#-----------------------------------------------------------------------
#--------------------------------------------
#-------------------------
# Boost Model

x_boost_train = x_train.tolist()
y_boost_train = y_train.tolist()



TP = 0
FP = 0
TN = 0
FN = 0
FP_x_train  = []
FP_y_train  = []
FP_index_train  = []
FN_x_train  = []
FN_y_train  = []
FN_index_train  = []
y_train1 = y_train.tolist()
pred1 = pred_train_clf.tolist()

for i in range(len(pred1)): 
    if y_train1[i]==pred1[i]==1:
      TP += 1
    elif pred1[i]==1 and y_train1[i]!=pred1[i]:
      FP += 1
      FP_x_train.append(x_train.loc[y_train.index[i]])
      FP_y_train.append(y_train.loc[y_train.index[i]])
      FP_index_train.append(y_train.index[i])
    elif y_train1[i]==pred1[i]==-1:
      TN += 1
    elif pred1[i]==-1 and y_train1[i]!=pred1[i]:
      FN += 1
      FN_x_train.append(x_train.loc[y_train.index[i]])
      FN_y_train.append(y_train.loc[y_train.index[i]])
      FN_index_train.append(y_train.index[i])



FP_FN_index_train = []
FP_FN_index_train = sorted(FP_index_train + FN_index_train)
cnt = 0
for j in range(len(FP_FN_index_train)):
    x_boost_train[FP_FN_index_train[j]+ cnt]= ",".join([x_boost_train[FP_FN_index_train[j]+ cnt], x_boost_train[FP_FN_index_train[j]+ cnt], x_boost_train[FP_FN_index_train[j]+ cnt]])  
    
    x_boost_train.insert(FP_FN_index_train[j]+ cnt, x_boost_train[FP_FN_index_train[j]+ cnt])
    y_boost_train.insert(FP_FN_index_train[j]+ cnt, y_boost_train[FP_FN_index_train[j]+ cnt])
    
    x_boost_train.insert(FP_FN_index_train[j]+ cnt, x_boost_train[FP_FN_index_train[j]+ cnt])
    y_boost_train.insert(FP_FN_index_train[j]+ cnt, y_boost_train[FP_FN_index_train[j]+ cnt])
    
    cnt+=2   



x_boost_train = pd.Series(x_boost_train)
y_boost_train = pd.Series(y_boost_train)




MAX_SENT_LENGTH = 300
MAX_SENTS = 2
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100



x_train_sent = []
y_train_sent = []

for idx in range(x_boost_train.shape[0]):
    sentences = tokenize.sent_tokenize(str(x_boost_train[idx]))
    x_train_sent.append(sentences)
    y_train_sent.append(y_boost_train[idx])
    



x_val_sent = []
y_val_sent = []

for iidx in range(x_val.shape[0]):
    sentences = tokenize.sent_tokenize(x_val[iidx])
    x_val_sent.append(sentences)
    y_val_sent.append(y_val[iidx])
     
    
    

x_test_sent = []
y_test_sent = []

for iiidx in range(x_test.shape[0]):
    sentences = tokenize.sent_tokenize(x_test[iiidx])
    x_test_sent.append(sentences)
    y_test_sent.append(y_test[iiidx])
     
    

    
Tweets_Tweets_test= []
Tweets_trainlist = x_train.tolist()
Tweets_vallist = x_val.tolist()
Tweets_Tweets_test = sorted(Tweets_trainlist + Tweets_vallist)    
Tweets_Tweets_test = pd.Series(Tweets_Tweets_test)    
    
 
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(Tweets_Tweets_test)

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))




data_train = np.zeros((len(x_train_sent), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(x_train_sent):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data_train[i, j, k] = tokenizer.word_index[word]
                    k = k + 1




countsss_1 = 0
data_val = np.zeros((len(x_val_sent), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(x_val_sent):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word in word_index:
                    countsss_1 += 1
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data_val[i, j, k] = tokenizer.word_index[word]
                        k = k + 1







countss_1 = 0
data_test = np.zeros((len(x_test_sent), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(x_test_sent):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word in word_index:
                    countss_1 += 1
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data_test[i, j, k] = tokenizer.word_index[word]
                        k = k + 1









print(pd.Series(y_train_sent).value_counts())



GLOVE_DIR = "E:/FakeNews/FakeNews-code/keras/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))



countss = 0
embedding_matrix = np.random.random((50000 + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < 50001:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            countss += 1
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector



# building Hierachical Attention network

embedding_layer = Embedding(50000 + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # return mask
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])




y_train_sent = y_boost_train
y_train_sent.value_counts
y_train_sent.value_counts()
y_train_sent = y_train_sent.replace(-1,0)
y_train_sent = to_categorical(y_train_sent)
print(y_train_sent.sum(axis=0))


y_val_sent = y_val
y_val_sent.value_counts
y_val_sent.value_counts()
y_val_sent = y_val_sent.replace(-1,0)
y_val_sent = to_categorical(np.asarray(y_val_sent))
print(y_val_sent.sum(axis=0))



history = model.fit(data_train, y_train_sent, validation_data=(data_val, y_val_sent),
          nb_epoch=2, batch_size=50)



from matplotlib import pyplot

plt.figure()
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='Train_Boost')
pyplot.plot(history.history['val_loss'], label='Val_Boost')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='Train_Boost')
pyplot.plot(history.history['val_acc'], label='Val_Boost')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend()

pyplot.show()





y_test_sent = y_test
y_test_sent.value_counts
y_test_sent.value_counts()
y_test_sent = y_test_sent.replace(-1,0)
y_test_sent = to_categorical(np.asarray(y_test_sent))
print(y_test_sent.sum(axis=0))



# predict probabilities for train set
yhat_probs_train = model.predict(data_train, verbose=1)
yhat_classes_train = yhat_probs_train.argmax(axis=-1)

# predict probabilities for test set
yhat_probs_test = model.predict(data_test, verbose=1)
yhat_classes_test = yhat_probs_test.argmax(axis=-1)

train_boost_loss, train_boost_acc = model.evaluate(data_train, y_train_sent, verbose=1)
test_boost_loss, test_boost_acc = model.evaluate(data_test, y_test_sent, verbose=1)

#---------------------------------------------------------

LSS_CLF = 1 - score_train_clf
LSS_BOOST = 1 - train_boost_acc


ALPHA_CLF = (0.5) * np.log((1-LSS_CLF)/LSS_CLF)
ALPHA_BOOST = (0.5) * np.log((1-LSS_BOOST)/LSS_BOOST)

print('LSS CLF:',LSS_CLF, 'LSS_Boost:',LSS_BOOST)
print('ALPHA CLF:',ALPHA_CLF, 'ALPHA_Boost:',ALPHA_BOOST)

if (((score_train_clf > train_boost_acc) and (score_test_clf < test_boost_acc)) or ((score_train_clf < train_boost_acc) and (score_test_clf > test_boost_acc))):
    temp = ALPHA_CLF
    ALPHA_CLF = ALPHA_BOOST
    ALPHA_BOOST = temp

pred_BOOST = pd.Series(yhat_classes_test)
pred_BOOST = pred_BOOST.replace(0,-1)

Hybrid = []
for L in range(len(y_test)):
    Hybrid.append(np.sign((ALPHA_CLF * pred_test_clf[L]) + (ALPHA_BOOST * pred_BOOST[L])))
   



y_train_BOOST = y_boost_train
y_train_BOOST = y_train_BOOST.replace(-1,0)

y_test_BOOST = y_test
y_test_BOOST = y_test_BOOST.replace(-1,0)



#confusion matrix train_BOOST
#classification_report train_BOOST
score_train_BOOST = metrics.accuracy_score(y_train_BOOST, yhat_classes_train)
#train_boost_loss, train_boost_acc = model.evaluate(data_train, y_train_sent, verbose=1)
kappa_train_BOOST = metrics.cohen_kappa_score(y_train_BOOST, yhat_classes_train)
auc_train_BOOST = metrics.roc_auc_score(y_train_sent, yhat_probs_train)
matrix_train_BOOST = metrics.confusion_matrix(y_train_BOOST, yhat_classes_train)

plt.figure()
cm = metrics.confusion_matrix(y_train_BOOST, yhat_classes_train, labels=[1, 0])
plot_confusion_matrix(cm, classes=['FAKE_Train_BOOST', 'REAL_Train_BOOST'])




#confusion matrix test_BOOST
#classification_report test_BOOST
score_test_BOOST = metrics.accuracy_score(y_test_BOOST, yhat_classes_test)
#test_boost_loss, test_boost_acc = model.evaluate(data_test, y_test_sent, verbose=1)
kappa_test_BOOST = metrics.cohen_kappa_score(y_test_BOOST, yhat_classes_test)
auc_test_BOOST = metrics.roc_auc_score(y_test_sent, yhat_probs_test)
matrix_test_BOOST = metrics.confusion_matrix(y_test_BOOST, yhat_classes_test)

plt.figure()
cm = metrics.confusion_matrix(y_test_BOOST, yhat_classes_test, labels=[1, 0])
plot_confusion_matrix(cm, classes=['FAKE_Test_BOOST', 'REAL_Test_BOOST'])





Hybrid = pd.Series(Hybrid)
y_Hybrid_sent = Hybrid
y_Hybrid_sent.value_counts
y_Hybrid_sent.value_counts()
y_Hybrid_sent = y_Hybrid_sent.replace(-1,0)
y_Hybrid_sent = to_categorical(np.asarray(y_Hybrid_sent))
print(y_Hybrid_sent.sum(axis=0))



#confusion matrix test_Hybrid
#classification_report test_Hybrid
score_test_Hybrid = metrics.accuracy_score(y_test, Hybrid)
#test_Hybrid_loss, test_Hybrid_acc = model.evaluate(data_test, y_Hybrid_sent, verbose=1)
kappa_test_Hybrid = metrics.cohen_kappa_score(y_test, Hybrid)
matrix_test_Hybrid = metrics.confusion_matrix(y_test, Hybrid)

plt.figure()
cm = metrics.confusion_matrix(y_test, Hybrid, labels=[1, -1])
plot_confusion_matrix(cm, classes=['FAKE_Test_Hybrid', 'REAL_Test_Hybrid'])




print("\n\n\n\n\n\n\n\n",
      "\n\n classification_report for BOOST_train Model\n\n",classification_report(y_train_BOOST, yhat_classes_train),
      "\n\n Accuracy_BOOST_train:   %0.3f" % score_train_BOOST,
      "\n\n train_BOOST_acc: %.3f, train_BOOST_loss: %.3f" %(train_boost_acc, train_boost_loss),
      "\n\n Cohens kappa_train_BOOST: %f" % kappa_train_BOOST,
      "\n\n ROC AUC of Train_BOOST : %f" % auc_train_BOOST,
      "\n\n",matrix_train_BOOST,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for BOOST_test Model\n\n",classification_report(y_test_BOOST, yhat_classes_test),
      "\n\n Accuracy_BOOST_test:   %0.3f" % score_test_BOOST,
      "\n\n test_BOOST_acc: %.3f, test_BOOST_loss: %.3f" % (test_boost_acc, test_boost_loss),
      "\n\n Cohens kappa_test_BOOST: %f" % kappa_test_BOOST,
      "\n\n ROC AUC  of Test_BOOST : %f" % auc_test_BOOST,
      "\n\n",matrix_test_BOOST,
      "\n\n\n",
      "\n\n\n",
      "\n\n\n",
      "\n\n classification_report for Hybrid_test Model\n\n",classification_report(y_test, Hybrid),
      "\n\n Accuracy_Hybrid_test:   %0.3f" % score_test_Hybrid,
#      "\n\n test_Hybrid_acc: %.3f, test_Hybrid_loss: %.3f" % (test_Hybrid_acc, test_Hybrid_loss),
      "\n\n Cohens kappa_test_Hybrid: %f" % kappa_test_Hybrid,
      "\n\n",matrix_test_Hybrid,
      "\n\n\n\n\n\n\n\n"
      )
      
#-----------------------------------------------------------------------
#--------------------------------------------
#-------------------------
# Results


print("\n\n\n\n\n",
      "**********************************",
      "\n ************* CLF ****************",
      "\n **********************************",
      "\n\n classification_report for CLF_train Model\n\n",classification_report(y_train, pred_train_clf),
      "\n\n Accuracy_CLF_train:   %0.3f" % score_train_clf,
      "\n\n Cohens kappa_train_clf: %f" % kappa_train_clf,
      "\n\n",matrix_train_clf,
      "\n\n",
      "-------------------------",
      "\n\n classification_report for CLF_test Model\n\n",classification_report(y_test, pred_test_clf),
      "\n\n Accuracy_CLF_test:   %0.3f" % score_test_clf,
      "\n\n Cohens kappa_test_clf: %f" % kappa_test_clf,
      "\n\n",matrix_test_clf,
      "\n\n\n\n\n",
      "**********************************",
      "\n ************* HAN ****************",
      "\n **********************************",
      "\n\n classification_report for HAN_train Model\n\n",classification_report(y_train_han, yhat_classes_train_han),
      "\n\n Accuracy_HAN_train:   %0.3f" % score_train_han,
      "\n\n train_HAN_acc: %.3f, train_HAN_loss: %.3f" %(train_han_acc, train_han_loss),
      "\n\n Cohens kappa_train_HAN: %f" % kappa_train_han,
      "\n\n ROC AUC of Train_HAN : %f" % auc_train_han,
      "\n\n",matrix_train_han,
      "\n\n",
      "-------------------------",
      "\n\n classification_report for HAN_test Model\n\n",classification_report(y_test_han, yhat_classes_test_han),
      "\n\n Accuracy_HAN_test:   %0.3f" % score_test_han,
      "\n\n test_HAN_acc: %.3f, test_HAN_loss: %.3f" % (test_han_acc, test_han_loss),
      "\n\n Cohens kappa_test_HAN: %f" % kappa_test_han,
      "\n\n ROC AUC  of Test_HAN : %f" % auc_test_han,
      "\n\n",matrix_test_han,
      "\n\n\n\n\n",
      "**********************************",
      "\n ************* BOOST ****************",
      "\n **********************************",
      "\n\n classification_report for BOOST_train Model\n\n",classification_report(y_train_BOOST, yhat_classes_train),
      "\n\n Accuracy_BOOST_train:   %0.3f" % score_train_BOOST,
      "\n\n train_BOOST_acc: %.3f, train_BOOST_loss: %.3f" %(train_boost_acc, train_boost_loss),
      "\n\n Cohens kappa_train_BOOST: %f" % kappa_train_BOOST,
      "\n\n ROC AUC of Train_BOOST : %f" % auc_train_BOOST,
      "\n\n",matrix_train_BOOST,
      "\n\n",
      "-------------------------",
      "\n\n classification_report for BOOST_test Model\n\n",classification_report(y_test_BOOST, yhat_classes_test),
      "\n\n Accuracy_BOOST_test:   %0.3f" % score_test_BOOST,
      "\n\n test_BOOST_acc: %.3f, test_BOOST_loss: %.3f" % (test_boost_acc, test_boost_loss),
      "\n\n Cohens kappa_test_BOOST: %f" % kappa_test_BOOST,
      "\n\n ROC AUC  of Test_BOOST : %f" % auc_test_BOOST,
      "\n\n",matrix_test_BOOST,
      "\n\n",
      "-------------------------",
      "\n\n classification_report for Hybrid_test Model\n\n",classification_report(y_test, Hybrid),
      "\n\n Accuracy_Hybrid_test:   %0.3f" % score_test_Hybrid,
#      "\n\n test_Hybrid_acc: %.3f, test_Hybrid_loss: %.3f" % (test_Hybrid_acc, test_Hybrid_loss),
      "\n\n Cohens kappa_test_Hybrid: %f" % kappa_test_Hybrid,
      "\n\n",matrix_test_Hybrid,
      "\n\n\n\n\n",
      )
   
