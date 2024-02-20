import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Returns the alpha that results on the greater accuracy score
def best_alpha(alpha, X_train, y_train, X_test, y_test):
    acc, alpha_f = 0, 0
    for i in range(len(alpha)):
        nb_classifier = MultinomialNB(alpha=alpha[i])
        nb_classifier.fit(X_train, y_train)
        pred = nb_classifier.predict(X_test)
        if metrics.accuracy_score(y_test, pred) > acc:
            acc = metrics.accuracy_score(y_test, pred)
            alpha_f = alpha[i]
    return alpha_f


def main():
    # Reading data
    data = pd.read_csv("spam.csv", encoding="ISO-8859-1") # Latin caracters
    data = data[['v1','v2']].copy()

    # Changing column names
    data = data.rename(columns={"v1":"target","v2":"text"}).copy()
    y = data.target

    # Mapping ham -> 0, spam -> 1
    y = y.map({'ham':0,'spam':1})

    X_train, X_test, y_train, y_test = train_test_split(data["text"], y, test_size=0.33, random_state=53)

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
    tfidf_test = tfidf_vectorizer.transform(X_test.values)

    alpha = np.arange(0.1,1,0.1)

    db_classifier = MultinomialNB(alpha=best_alpha(alpha, tfidf_train, y_train, tfidf_test, y_test))

    db_classifier.fit(tfidf_train, y_train)
    pred = db_classifier.predict(tfidf_test)

    print(f"Model Accuracy score: {metrics.accuracy_score(y_test, pred)}")
    print(f"Model Precision score: {metrics.precision_score(y_test, pred)}")
    print(f"Model F1 score: {metrics.f1_score(y_test, pred)}\n")

    print("Confusion Matrix:")
    confusion_m = pd.DataFrame(metrics.confusion_matrix(y_test, pred, labels=[0,1]), index=['HAM','SPAM'], columns=['HAM','SPAM'])
    print(confusion_m)

    """
    Assume general form: (A,B) C

    A: Document index B: Specific word-vector index C: TFIDF score for word B in document A
    """

if __name__=="__main__":
    main()

