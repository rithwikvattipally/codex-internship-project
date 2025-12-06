import pandas as pd

df = pd.read_csv("spam.csv", encoding='latin-1')
df.head()

df = df[['v1', 'v2']]
df.columns = ['label', 'message']   # rename columns
df.head()

df.info()
df['label'].value_counts()

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='label', data=df)
plt.title("Ham vs Spam Count")
plt.show()

print("Spam example:\n", df[df['label']=='spam']['message'].iloc[0])
print("\nHam example:\n", df[df['label']=='ham']['message'].iloc[0])

X = df['message']         # text data
y = df['label_num']       # 0 = ham, 1 = spam

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)   # learn vocab + transform
X_test_tfidf = tfidf.transform(X_test)        # only transform

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

def predict_message(msg):
    msg_tfidf = tfidf.transform([msg])
    pred = model.predict(msg_tfidf)[0]
    return "Spam" if pred == 1 else "Ham"

print(predict_message("Congratulations! You have won a lottery of $1,000,000. Call now!"))
print(predict_message("Hey, are we meeting at 5 pm today?"))

