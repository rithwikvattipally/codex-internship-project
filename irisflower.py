import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier   # you can also try others
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = iris.data          # features
y = iris.target        # labels (0,1,2)
feature_names = iris.feature_names
target_names = iris.target_names

print(feature_names)
print(target_names)

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df.head()

df.shape
df['species'].value_counts()

df.describe()

df.hist(figsize=(8,6))
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='species')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)   # or X_train_scaled if you used scaler

y_pred = knn.predict(X_test)    # or X_test_scaled 
print(y_pred[:10])
print(y_test[:10])

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_test, y_pred, target_names=target_names))

sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # example values
pred_class = knn.predict(sample)[0]
print("Predicted species:", target_names[pred_class])
