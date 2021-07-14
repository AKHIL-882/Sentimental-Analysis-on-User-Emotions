import pandas as pd
import numpy as np
import matplotlib.pyplot as plt        #Visualisation
import seaborn as sns  

import streamlit as st
import nltk
nltk.download('stopwords')


from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score






st.write("-------------------------------------------------------\nDisplaying Train Data\n-------------------------------------------------------")
train = pd.read_csv("train_new.csv")
st.dataframe(train)


st.write("-------------------------------------------------------\nChecking Null values\n-------------------------------------------------------")
st.write(" " ,train.isnull().sum())
st.write("-------------------------------------------------------\nShape of the dataset\n-------------------------------------------------------")
st.write(" ",train.shape)
st.write("-------------------------------------------------------\nInformation of dataset\n-------------------------------------------------------")
st.write("", train.info())
st.write("-------------------------------------------------------\nDescription of the dataset\n-------------------------------------------------------")
st.write("",train.describe())
st.write("-------------------------------------------------------\nChecking duplication records\n-------------------------------------------------------")
st.write(sum(train.duplicated()))
st.write("-------------------------------------------------------\nDroping the NaN values\n-------------------------------------------------------")

st.write(train.dropna(axis='columns'))
st.write(train.dropna(how='all'))
st.write("-------------------------------------------------------\nChecking for Null\n-------------------------------------------------------")
st.write(train.isnull().sum())





test = pd.read_csv("test_new.csv")
st.dataframe(test)



st.write("-------------------------------------------------------\nDisplaying Test Data\n-------------------------------------------------------")

st.write("-------------------------------------------------------\nChecking Null values\n-------------------------------------------------------")
st.write(" " ,test.isnull().sum())
st.write("-------------------------------------------------------\nShape of the dataset\n-------------------------------------------------------")
st.write(" ",test.shape)
st.write("-------------------------------------------------------\nInformation of dataset\n-------------------------------------------------------")
st.write("", test.info())
st.write("-------------------------------------------------------\nDescription of the dataset\n-------------------------------------------------------")
st.write("",test.describe())
st.write("-------------------------------------------------------\nChecking duplication records\n-------------------------------------------------------")
st.write(sum(test.duplicated()))
st.write("-------------------------------------------------------\nDroping the NaN values\n-------------------------------------------------------")

st.write(test.dropna(axis='columns'))
st.write(test.dropna(how='all'))
st.write("-------------------------------------------------------\nChecking for Null\n-------------------------------------------------------")
st.write(test.isnull().sum())




import re

# helper function
def clean_text(text):
    te = str(text).encode('ascii','ignore').decode('UTF-8')
    te = re.sub(r'@[\w]+', '', te)
    te = re.sub(r'https?://t.co/[\w]+', '', te)
    te = re.sub(r'#', '', te)
    te = re.sub(r"RT @[\w]+:",'',te)
    te = re.sub(r"RT @[\w]+:",'',te)
    te = re.sub(r" RT ",'',te)
    te = re.sub(r"https://[\w]+.[\w]+/[\w]+",'',te)
    te = re.sub(r"[][]",'',te)
    te = re.sub(r"&amp","and", te)
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", te)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text



 # Transform each text into a vector of word counts
vectorizer = TfidfVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1, 2))

training_features = vectorizer.fit_transform(train.text)
st.write(training_features)




# Transform each text into a vector of word counts
vectorizer2 = TfidfVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1, 2))

testing_features = vectorizer2.fit_transform(test.text)
st.write(testing_features)




from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# extract the labels from the train data
y = train.target.values

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(train.text.values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.3, shuffle=True)


# remove special characters using the regular expression library
import re

#set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")




import preprocessor as p

# custum function to clean the dataset (combining tweet_preprocessor and reguar expression)
def clean_tweets(df):
  tempArr = []
  for line in df:
    # send to tweet_processor
    tmpL = p.clean(line)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr





train_tweet = clean_tweets(train["text"])
train_tweet = pd.DataFrame(train_tweet)
# append cleaned tweets to the training data
train["clean_tweet"] = train_tweet
# compare the cleaned and uncleaned tweets
train.head()




# clean the test data and append the cleaned tweets to the test data
test_text = clean_tweets(test["text"])
test_text = pd.DataFrame(test_text)
# append cleaned tweets to the training data
test["clean_text"] = test_text

# compare the cleaned and uncleaned tweets
test.head()



from sklearn.model_selection import train_test_split

# extract the labels from the train data
y = train.target.values

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(train.clean_tweet.values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.3, shuffle=True)




documents = ["This is Import Data's Youtube channel",
             "Data science is my passion and it is fun!",
             "Please subscribe to my channel"]

# initializing the countvectorizer
vectorizer = CountVectorizer()

# tokenize and make the document into a matrix
document_term_matrix = vectorizer.fit_transform(documents)

# check the result
pd.DataFrame(document_term_matrix.toarray(), columns = vectorizer.get_feature_names())




from sklearn.feature_extraction.text import CountVectorizer

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)



from sklearn import svm
# classify using support vector classifier
svm = svm.SVC(kernel = 'linear', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(x_train_vec, y_train).predict_proba(x_test_vec)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_test_vec)


st.write("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%')



data = [train['clean_tweet'], train["target"]]

headers = ["clean_text", "target"]

train_new = pd.concat(data, axis=1, keys=headers)
train_new.head()


train, test = train_test_split(train_new, test_size=0.2, random_state=1)
X_train = train['clean_text'].values
X_test = test['clean_text'].values
y_train = train['target']
y_test = test['target']



def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

en_stopwords = set(stopwords.words("english")) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)



kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)




np.random.seed(1)

pipeline_svm = make_pipeline(vectorizer, 
                            SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)



def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'Accuracy': auc, 'F1-Score': f1, 'Accuracy': acc, 'Precision': prec, 'Recall': rec}
    return result
    
report_results(grid_svm.best_estimator_, X_test, y_test)




user_input = st.text_input("Enter the string")
res = grid_svm.predict([user_input])
if(res[0] == 1):
  st.write("\n Positive \n")
else:
  st.write("\n Negative \n")