# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:34:46 2019

@author: agabh
"""

import nltk
import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
import string
import copy
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer, PorterStemmer
from collections import Counter
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy import stats
from scipy.stats import uniform, randint
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils


import re
import math
                               
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

punct = list(string.punctuation)
punct = [unicode(x) for x in punct]

#changing current working directory 
os.chdir(r"C:\Users\agabh\Desktop\Hackathon_2")
print(os.getcwd())

def wordnet_tag(treebank_tag):
    
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('V'):
        return 'v'
    else:
        return 'n'

data = pd.read_excel(r"C:\Users\agabh\Desktop\Hackathon_2\Participants_Data_News_category\Data_Train.xlsx")
score_data = pd.read_excel(r"Participants_Data_News_category\Data_Test.xlsx")
bollywood_data = pd.read_excel(r"bollywood2.xlsx")
hollywood_data = pd.read_csv(r"tmdb_5000_credits.csv")
bollywood_actors_data = pd.read_csv(r"BollywoodActorRanking.csv")
bollywood_directors_data = pd.read_csv(r'BollywoodDirectorRanking.csv')


bollywood_movies = bollywood_data['Title'].tolist()
bollywood_movies = [x for x in bollywood_movies if type(x) == unicode]
bollywood_movies = [x.lower() for x in bollywood_movies]

bollywood_dir = bollywood_data['Director'].tolist()
bollywood_dir = [x for x in bollywood_dir if type(x) == unicode]
bollywood_dir = [x.lower() for x in bollywood_dir]
bollywood_dir2 = []
for dir in bollywood_dir:
    dirs = dir.split(', ')
    bollywood_dir2.extend(dirs)
    
    
bollywood_cast = bollywood_data['Cast'].tolist()
bollywood_cast = [x for x in bollywood_cast if type(x) == unicode]
pattern = re.compile(r"([a-z])([A-Z])")
bollywood_cast2 = []
for cast in bollywood_cast:
    cast2 = pattern.sub(r'\1,\2', cast)
    cast2 = cast2.split(',')
    bollywood_cast2.extend(cast2)
bollywood_cast3 = list(set([x.lower() for x in bollywood_cast2 if len(x.split(' ')) > 1]))

bollywood_actors = bollywood_actors_data['actorName'].tolist()
bollywood_actors = [unicode(x.lower()) for x in bollywood_actors]

bollywood_cast3.extend(bollywood_actors)
bollywood_cast3 = list(set(bollywood_cast3))

bollywood_dir3 = bollywood_directors_data['directorName'].tolist()
bollywood_dir3 = [unicode(x) for x in bollywood_dir3]
bollywood_dir3 = [x for x in bollywood_dir3 if str(x) != 'nan']
bollywood_dir3 = list(set([x.lower() for x in bollywood_dir3]))

#male_names_data = pd.read_csv(r"Indian-Male-Names.csv")
#female_names_data = pd.read_csv(r"Indian-Female-Names.csv")
#
#male_names = male_names_data['name'].tolist()
#female_names = female_names_data['name'].tolist()
#
#male_names = [x for x in male_names if str(x) != 'nan']
#female_names = [x for x in female_names if str(x) != 'nan']
#
#male_names2 = []
#female_names2 = []
#
#for name in  male_names:
#    tokens = word_tokenize(name)
#    tokens = [x for x in tokens if x.isalpha()]
#    tokens = [unicode(x) for x in tokens]
#    for token in tokens:
#        male_names2.append(token)
#        
#for name in  female_names:
#    tokens = word_tokenize(name)
#    tokens = [x for x in tokens if x.isalpha()]
#    tokens = [unicode(x) for x in tokens]
#    for token in tokens:
#        female_names2.append(token)
#        
#names = male_names2[:]
#names.extend(female_names2)
#names = set(names)


#tokenization
str_list = data['STORY'].tolist()
str_labels = data['SECTION'].tolist()
str_labels = [int(x) for x in str_labels]

data_pol = data['STORY'][data.SECTION == 0].tolist()
data_tech = data['STORY'][data.SECTION == 1].tolist()
data_ent = data['STORY'][data.SECTION == 2].tolist()
data_biz = data['STORY'][data.SECTION == 3].tolist()

score_data_str = score_data['STORY'].tolist()
#str_list2 = [x.encode('utf-8') for x in str_list]

pattern_hash = re.compile(r"([#])([A-z])")
pattern_tweet =  re.compile(r"([A-z])([\.]+)")
pattern_tag = re.compile(r"([@])([A-z])")
pattern_perc = re.compile(r"([\%])")

def regex_process(str_list):
    str_list2 = []
    for string1 in str_list:
        string2 = pattern_hash.sub(r' hashtag \2',string1)
        string3 = pattern_tweet.sub(r'\1', string2)
        string4 = pattern_tag.sub(r' user_mention \2', string3)
        string5 = pattern_perc.sub(r' percent', string4)
        str_list2.append(string5)
    return str_list2

def pre_process(str_list):
    tokens = [word_tokenize(x) for x in str_list]
    tokens2 = []
    docs = []
    #1. lower case
    #remove punctuations
    for token in tokens:
        token = [x for x in token if x not in punct]
        token = [x for x in token if x.isalpha()]
        
        token2 = []
        pos_tags = nltk.pos_tag(token)
        for tags in pos_tags:
            tag = tags[1]
            tag_2 = wordnet_tag(tag)
            lemma = lemmatizer.lemmatize(tags[0], tag_2)
            token2.append(lemma)
        
        token = [x.lower() for x in token2]
        token = [stemmer.stem(x) for x in token]
        token = [x for x in token if x not in stop_words]
        token = [x for x in token if len(x) > 1]
        doc = ' '.join(token)
        tokens2.append(token)
        docs.append(doc)
        
    return tokens2, docs

str_list_copy = []
for strings in str_list:
    for cast in bollywood_cast3:
        strings = re.sub(cast, 'bollywood', strings.lower())
    str_list_copy.append(strings)
    
score_data_str_copy = []
iter = 1
for strings in score_data_str:
    print(iter)
    for cast in bollywood_cast3:
        strings = re.sub(cast, 'bollywood', strings.lower())
    score_data_str_copy.append(strings) 
    iter += 1

str_list_regp = regex_process(str_list_copy)
score_data_str_regp = regex_process(score_data_str)

train_data_tot_tokens, train_data_tot_docs = pre_process(str_list_regp)
score_data_tot_tokens, score_data_tot_docs = pre_process(score_data_str_regp)


#def token_distro(tokens):
#    token_set = []
#    for i in tokens:
#        for j in i:
#            token_set.append(j)    
#    return list(set(token_set))
#
#token_set = token_distro(train_data_tot_tokens)
#
#def token_distro_tag(str_list, tag):
#    tokens = [word_tokenize(x) for x in str_list]
#    tokens2 = []
#    for token in tokens:
#        token = [x for x in token if x not in punct]
#        token = [x for x in token if x.isalpha()]
#        
#        pos_tags = nltk.pos_tag(token)
#        for tags in pos_tags:
#            if tags[1] == tag :
#                tokens2.append(tags[0])
#    
#    return tokens2
#
#bol_token_NNP = token_distro_tag(data_ent, "NNP")        
#bol_token__NNP_distro = Counter(bol_token_NNP).most_common()
#
#pol_token_NNP = token_distro_tag(data_pol, "NNP")        
#pol_token_NNP_distro = Counter(pol_token_NNP).most_common()
#
#biz_token_NNP = token_distro_tag(data_biz, "NNP")        
#biz_token_NNP_distro = Counter(biz_token_NNP).most_common()
#
#tech_token_NNP = token_distro_tag(data_tech, "NNP")        
#tech_token_NNP_distro = Counter(tech_token_NNP).most_common()
#
#score_dt_token_NNP = token_distro_tag(score_data_str, "NNP")
#score_dt_token_NNP_distro = Counter(score_dt_token_NNP).most_common()
#
#
#tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, 
#                                 max_df=0.7, stop_words=stopwords.words('english'))
#X = tfidfconverter.fit_transform(train_data_tot_docs).toarray()
#vocab = tfidfconverter.get_feature_names()
#
#
#tfidfconverter2 = TfidfVectorizer(stop_words=stopwords.words('english'), vocabulary = vocab)
#X_score = tfidfconverter2.fit_transform(score_data_tot_docs).toarray()
#
#X_train, X_test, Y_train, Y_test =  train_test_split(X, 
#                                                     str_labels,
#                                                     test_size = 0.25, 
#                                                     random_state = 128900)
#
#svm_model_linear = SVC(kernel = 'linear', C = 1.2).fit(X_train, Y_train) 
#
#svm_predictions_test = svm_model_linear.predict(X_test) 
#accuracy_test = svm_model_linear.score(X_test, Y_test) 
#cm_test = confusion_matrix(Y_test, svm_predictions_test)
#
#svm_predictions_train = svm_model_linear.predict(X_train) 
#accuracy_train = svm_model_linear.score(X_train, Y_train) 
#cm_train = confusion_matrix(Y_train, svm_predictions_train)
#
#svm_predictions_score = svm_model_linear.predict(X_score)
#score_data['SECTION'] = svm_predictions_score
#score_data.to_excel("Scores.xlsx", index = False)
#
#
#vocab_range = np.arange(2150,4050,100)
#min_df_range = np.arange(3,5,1)
#max_df_range = np.arange(0.4, 0.6,0.1)
#C_range = np.arange(1,1.6,0.1)
#
#column = ['vocab_len',
#           'min_df',
#           'max_df',
#           'test_accuracy',
#           'train_accuracy']
#
#df_params = pd.DataFrame(columns = column)
#
#iteration = 1
#for vocab_len in vocab_range:
#    for minm_df in min_df_range:
#        for maxm_df in max_df_range:
#            tfidfconverter = TfidfVectorizer(max_features=vocab_len, min_df=minm_df, 
#                                 max_df=maxm_df, stop_words=stopwords.words('english'))
#            X = tfidfconverter.fit_transform(train_data_tot_docs).toarray()
#            vocab = tfidfconverter.get_feature_names()
#            
#            tfidfconverter2 = TfidfVectorizer(stop_words=stopwords.words('english'), vocabulary = vocab)
#            X_score = tfidfconverter2.fit_transform(score_data_tot_docs).toarray()
#            X_train, X_test, Y_train, Y_test =  train_test_split(X, 
#                                                     str_labels,
#                                                     test_size = 0.25, 
#                                                     random_state = 128900)
#
#            svm_model_linear = SVC(kernel = 'linear', C = 1.2).fit(X_train, Y_train) 
#            
#            svm_predictions_test = svm_model_linear.predict(X_test) 
#            accuracy_test = svm_model_linear.score(X_test, Y_test) 
#            cm_test = confusion_matrix(Y_test, svm_predictions_test)
#            
#            svm_predictions_train = svm_model_linear.predict(X_train) 
#            accuracy_train = svm_model_linear.score(X_train, Y_train) 
#            cm_train = confusion_matrix(Y_train, svm_predictions_train)
#            
#            svm_predictions_score = svm_model_linear.predict(X_score)
#            score_data['SECTION'] = svm_predictions_score
#            score_data.to_excel("Scores.xlsx", index = False)
#            
#            print('iteration : %d' %iteration)
#            print('vocab_len : %d' %vocab_len)
#            print('minm_df : %d' %minm_df)
#            print('maxm_df : %d' %maxm_df)
#            print('test_accuracy : %d' %accuracy_test)
#            print('train_accuracy : %d' %accuracy_train)
#            
#            temp = list((vocab_len,
#                         minm_df,
#                         maxm_df,   
#                         accuracy_test,
#                         accuracy_train))
#            df_params.loc[len(df_params)] = temp
#            print(df_params)
#            iteration += 1
#            df_params.to_excel("params2.xlsx", index = False)
#            
#def score_data_fn(vocab_len, minm_df, maxm_df):
#    tfidfconverter = TfidfVectorizer(max_features=vocab_len, min_df=minm_df, 
#                                 max_df=maxm_df, stop_words=stopwords.words('english'))
#    X = tfidfconverter.fit_transform(train_data_tot_docs).toarray()
#    vocab = tfidfconverter.get_feature_names()
#            
#    tfidfconverter2 = TfidfVectorizer(stop_words=stopwords.words('english'), vocabulary = vocab)
#    X_score = tfidfconverter2.fit_transform(score_data_tot_docs).toarray()
#    
#    
#    X_train, X_test, Y_train, Y_test =  train_test_split(X, 
#                                                     str_labels,
#                                                     test_size = 0, 
#                                                     random_state = 128900)
#    print(vocab_len)
#    print(minm_df)
#    print(maxm_df)
#    print(len(X_test))
#
#    svm_model_linear = SVC(kernel = 'linear', C = 1.2).fit(X_train, Y_train)
#    
#    svm_predictions_score = svm_model_linear.predict(X_score)
#    score_data['SECTION'] = svm_predictions_score
#    score_data.to_excel("Scores.xlsx", index = False)
#    
#    return vocab
#
#vocab_u = score_data_fn(2350, 3,0.5)


def features(vocab_len, minm_df, maxm_df, split_ratio):
    tfidfconverter = TfidfVectorizer(max_features=vocab_len, min_df=minm_df, 
                                 max_df=maxm_df, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(train_data_tot_docs).toarray()
    vocab = tfidfconverter.get_feature_names()
            
    tfidfconverter2 = TfidfVectorizer(stop_words=stopwords.words('english'), vocabulary = vocab)
    X_score = tfidfconverter2.fit_transform(score_data_tot_docs).toarray()
    
    X_train, X_test, Y_train, Y_test =  train_test_split(X, 
                                                     str_labels,
                                                     test_size = split_ratio, 
                                                     random_state = 128900)
    return X_train, X_test, Y_train, Y_test, X_score, vocab

vocab_len = 3500
min_df = 3
max_df = 0.5
split_ratio = 0.0005
X_train, X_test, Y_train, Y_test, X_score, vocab = features(vocab_len, min_df, max_df, split_ratio)
    
def random_forest(X_train, Y_train, X_test, Y_test, X_score):
    clf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',random_state=128900,verbose = 1, n_jobs = 4)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

def multi_NB(X_train, Y_train, X_test, Y_test, X_score):
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

def svm(X_train, Y_train, X_test, Y_test, X_score):
    svm_model_linear = SVC(kernel = 'linear', C = 1.2).fit(X_train, Y_train)
    svm_predictions_score = svm_model_linear.predict(X_score)
    preds = svm_model_linear.predict(X_test)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(svm_model_linear, X_train, Y_train, verbose = 1, cv = 3)
    return svm_predictions_score, accuracy, report, cm, cv_score

def svm_l(X_train, Y_train, X_test, Y_test, X_score):
    clf = LinearSVC(random_state = 128900, tol = 1e-5)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

def nn(X_train, Y_train, X_test, Y_test, X_score):
    clf = MLPClassifier(hidden_layer_sizes=(150,10), random_state=128900, verbose = True, tol = 1e-6)
    clf.fit(X_train, Y_train)
    scores = clf.predict(X_score)
    preds = clf.predict(X_test)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

def shallow_nn(X_train, Y_train, X_test, Y_test, X_score):
    clf = MLPClassifier(hidden_layer_sizes=(300), random_state=128900, verbose = True, tol = 1e-6)
    clf.fit(X_train, Y_train)
    scores = clf.predict(X_score)
    preds = clf.predict(X_test)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

def xg_b(X_train, Y_train, X_test, Y_test, X_score):
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb_model.fit(X_train, Y_train)
    preds = xgb_model.predict(X_test)
    scores = xgb_model.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(xgb_model, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

def multi_logit(X_train, Y_train, X_test, Y_test, X_score):
    clf = LogisticRegression(random_state = 128900, solver='lbfgs',multi_class='multinomial')
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score 

def ridge(X_train, Y_train, X_test, Y_test, X_score):
    clf = RidgeClassifier()
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score 

def ridge_CV(X_train, Y_train, X_test, Y_test, X_score):
    clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score 

def QDA(X_train, Y_train, X_test, Y_test, X_score):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score 

def extra_trees(X_train, Y_train, X_test, Y_test, X_score):
    clf = ExtraTreesClassifier(n_estimators = 1000, criterion = 'entropy',random_state=128900,verbose = 1, n_jobs = 4)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    scores = clf.predict(X_score)
    accuracy = metrics.accuracy_score(preds, Y_test)
    report = classification_report(Y_test, preds)
    cm = confusion_matrix(Y_test, preds)
    cv_score = cross_val_score(clf, X_train, Y_train, verbose = 1, cv = 3)
    return scores, accuracy, report, cm, cv_score

scores_random_forest, acc_rf, rep_rf, cm_rf, cv_rf = random_forest(X_train, Y_train, X_test, Y_test, X_score)
scores_mNB, acc_mNB, rep_mNB, cm_mNB, cv_mNB = multi_NB(X_train, Y_train, X_test, Y_test, X_score)
scores_svm, acc_svm, rep_svm, cm_svm, cv_svm = svm(X_train, Y_train, X_test, Y_test, X_score)
scores_nn, acc_nn, rep_nn, cm_nn, cv_nn = nn(X_train, Y_train, X_test, Y_test, X_score)
#scores_xgb, acc_xgb, rep_xgb, cm_xgb, cv_xgb = xg_b(X_train, Y_train, X_test, Y_test, X_score)
scores_logit, acc_logit, rep_logit, cm_logit, cv_logit = multi_logit(X_train, Y_train, X_test, Y_test, X_score)
scores_ridge, acc_ridge, rep_ridge, cm_ridge, cv_ridge = ridge(X_train, Y_train, X_test, Y_test, X_score)
#scores_qda, acc_qda, rep_qda, cm_qda, cv_qda = QDA(X_train, Y_train, X_test, Y_test, X_score)
scores_ridge_cv, acc_ridge_cv, rep_ridge_cv, cm_ridge_cv, cv_ridge_cv = ridge_CV(X_train, Y_train, X_test, Y_test, X_score)
scores_svm_l, acc_svm_l, rep_svm_l, cm_svm_l, cv_svm_l = svm_l(X_train, Y_train, X_test, Y_test, X_score)
scores_et, acc_et, rep_et, cm_et, cv_et = extra_trees(X_train, Y_train, X_test, Y_test, X_score)
scores_s_nn, acc_s_nn, rep_s_nn, cm_s_nn, cv_s_nn = shallow_nn(X_train, Y_train, X_test, Y_test, X_score)

#individual score accuracies
scores_df = pd.DataFrame()
scores_df['SECTION'] = scores_s_nn
scores_df.to_excel("Scores_nn_3500.xlsx", index = False)

scores_df = pd.DataFrame()
scores_df['SECTION'] = scores_svm
scores_df.to_excel("Scores_svm_3500.xlsx", index = False)

scores_df = pd.DataFrame()
scores_df['SECTION'] = scores_mNB
scores_df.to_excel("Scores_mNB.xlsx", index = False)

scores_df = pd.DataFrame()
scores_df['SECTION'] = scores_nn
scores_df.to_excel("Scores_dnn_3500.xlsx", index = False)


#voting method
scores_m = np.array([scores_s_nn, scores_logit, scores_svm, scores_ridge, scores_mNB, scores_et, scores_nn])
scores_t = stats.mode(scores_m)
scores_f = np.array(scores_t[0])
scores_f = np.reshape(scores_f, ((scores_f.shape[1],1)))
scores_f = scores_f.flatten()
scores_df = pd.DataFrame()
scores_df['SECTION'] = scores_f
scores_df.to_excel("Scores13.xlsx", index = False)

