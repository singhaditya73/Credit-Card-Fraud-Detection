import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('fraudTrain.csv')
df_train.head(5)
df_test = pd.read_csv('fraudTest.csv')
df_test.head(5)

print(f"The shape of train set: {df_train.shape}")
print(f"Test shape of test set: {df_test.shape}")

df_train.info()

df_train.isnull().sum()

df_test.info()

df_test.isnull().sum()

def clean_data(clean):
     clean.drop(["Unnamed: 0",'cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],axis=1, inplace=True)
     clean.dropna()
     return clean

clean_data(df_train)

clean_data(df_test)

df_train.select_dtypes(include = ['object'])

encoder=LabelEncoder()
def encode(data):
    data['merchant']=encoder.fit_transform(data['merchant'])
    data["category"] = encoder.fit_transform(data["category"])
    data["gender"] = encoder.fit_transform(data["gender"])
    data["job"] = encoder.fit_transform(data["job"])
    return data

encode(df_train)

encode(df_test)

from matplotlib import pyplot as plt
exit_counts = df_train["is_fraud"].value_counts()
plt.figure(figsize=(7, 7))
plt.subplot(1, 2, 1)  
plt.pie(exit_counts, labels=["No", "YES"], autopct="%0.0f%%")
plt.title("is_fraud Counts")
plt.tight_layout() 
plt.show()

import seaborn as sns
pd.options.display.float_format = "{:,.2f}".format

corr_matrix = df_train.corr(method = 'pearson')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0

cmap = "mako"

sns.heatmap(corr_matrix, mask=mask, vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot_kws={"size": 9, "color": "black"}, square=True, cmap=cmap, annot=True)

x=df_train.drop(columns=['is_fraud'])
y=df_train['is_fraud']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = DecisionTreeClassifier()
from sklearn.metrics import accuracy_score

def acc_score(test, pred):
    acc_ = accuracy_score(test, pred)
    return acc_

def print_score(test, pred, model):

    print(f"Classifier: {model}")
    print(f"ACCURACY: {accuracy_score(test, pred)}")

model1.fit(x_train,y_train)
y_pred = model1.predict(x_test)
print_score(y_test, y_pred, "Logistic Regression")
model_list = []
acc_list = []

model_list.append(model1.__class__.__name__)
acc_list.append(round(acc_score(y_test, y_pred), 4))

model2.fit(x_train,y_train)
y_pred1 = model2.predict(x_test)
print_score(y_test,y_pred1,"Random Forest")
model_list.append(model2.__class__.__name__)
acc_list.append(round(acc_score(y_test, y_pred), 4))

model3.fit(x_train,y_train)
Y_Pred = model3.predict(x_test)
print_score(y_test, Y_Pred, "Decision Tree")
model_list.append(model3.__class__.__name__)
acc_list.append(round(acc_score(y_test, Y_Pred), 3))

#comparison and result 
model_results = pd.DataFrame({"Model": model_list,
                              "Accuracy_Score": acc_list,
                              })
model_results
