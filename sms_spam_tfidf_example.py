"""
SMS Spam Detection - TF-IDF + Multinomial Naive Bayes / SVM
Dataset: UCI "SMS Spam Collection" (5,574 messages, labels: ham / spam)

Quy trinh (tuong tu cac bai bao ve SMS spam tren UCI dataset):
  1. Doc du lieu           2. Tien xu ly (lowercase, bo ky tu, stopword, stemming)
  3. TF-IDF                4. Huan luyen MNB va LinearSVC
  5. Danh gia (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)

Cai dat:  pip install pandas scikit-learn nltk
"""
import io
import re
import urllib.request
import zipfile

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_URL = ("https://raw.githubusercontent.com/justmarkham/"
              "pycon-2016-tutorial/master/data/sms.tsv")
stemmer = PorterStemmer()


def load_data() -> pd.DataFrame:
    """Doc du lieu: file local 'SMSSpamCollection' -> UCI -> ban sao tren GitHub."""
    import os
    if os.path.exists("SMSSpamCollection"):
        df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                         names=["label", "text"])
    else:
        try:
            raw = urllib.request.urlopen(URL, timeout=30).read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                with z.open("SMSSpamCollection") as f:
                    df = pd.read_csv(f, sep="\t", header=None,
                                     names=["label", "text"])
        except Exception as e:  # UCI bi chan/loi mang -> dung ban sao du phong
            print("Khong tai duoc tu UCI (", e, ") -> dung ban sao GitHub")
            df = pd.read_csv(MIRROR_URL, sep="\t", header=None,
                             names=["label", "text"])
    df["y"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0
    return df


def preprocess(text: str) -> str:
    """Lowercase -> giu chu cai -> bo stopword -> stemming."""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(stemmer.stem(t) for t in tokens)


def main():
    df = load_data()
    print(df["label"].value_counts(), "\n")

    df["clean"] = df["text"].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(C=1.0),
    }
    for name, model in models.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        score = (model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba")
                 else model.decision_function(Xte))
        print(f"=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_test, pred), 4))
        print("ROC-AUC :", round(roc_auc_score(y_test, score), 4))
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=["ham", "spam"]), "\n")

    # Du doan tin nhan moi bang mo hinh SVM
    svm = models["Linear SVM"]
    samples = [
        "Congratulations! You won a $1000 prize. Call now to claim!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]
    for s in samples:
        label = svm.predict(vec.transform([preprocess(s)]))[0]
        print(("SPAM" if label else "HAM"), "->", s)


if __name__ == "__main__":
    main()
