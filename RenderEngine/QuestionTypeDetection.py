import logging
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


# Inspired from https://github.com/kartikn27/nlp-question-detection

class QuestionTypeDetection():

    def __init__(self, classification_type):
        self.classification_type = classification_type
        df = self.training_data()
        df = self.preprocessing(df)
        df = self.__label_encode(df)
        vectorizer_classifier = self.__create_classifier(df, self.classification_type)
        if vectorizer_classifier is not None:
            self.vectorizer = vectorizer_classifier['vectorizer']
            self.classifier = vectorizer_classifier['classifier']

    def preprocessing(self, df):
        df.rename(columns={0: 'text', 1: 'type'}, inplace=True)
        df['type'] = df['type'].str.strip()
        df['text'] = df['text'].apply(lambda x: x.lower())
        df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s\']', '', x)))
        return df[
            (df['type'] == 'what') | (df['type'] == 'who') | (df['type'] == 'when') | (
                    df['type'] == 'unknown') | (
                    df['type'] == 'affirmation')]

    def __label_encode(self, df):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(df['type'])
        df['label'] = list(self.label_encoder.transform(df['type']))
        return df

    def __create_classifier(self, df, classification_type):
        v = TfidfVectorizer(analyzer='word', lowercase=True)
        X = v.fit_transform(df['text'])
        X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.30)
        if classification_type == 'MNB':
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            print(classification_report(preds, y_test))
            return {'vectorizer': v, 'classifier': clf}
        elif classification_type == 'SVM':
            clf_svm = SVC(kernel='linear')
            clf_svm.fit(X_train, y_train)
            preds = clf_svm.predict(X_test)
            print(classification_report(preds, y_test))
            print('Accuracy is: ', clf_svm.score(X_test, y_test))
            return {'vectorizer': v, 'classifier': clf_svm}
        else:
            print(
                "Wrong classification type: \n Type 'MNB' - Multinomial Naive Bayes \n Type 'SVM' - Support Vector Machine")

    def training_data(self):
        return pd.read_csv('sample.txt', sep=',,,', header=None)

    # Return: Kind of question 'what', 'when', 'who'
    def predict(self, sentence):
        ex = self.vectorizer.transform([sentence])
        return list(self.label_encoder.inverse_transform(self.classifier.predict(ex)))[0]
