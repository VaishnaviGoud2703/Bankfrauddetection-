#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,f1_score,precision_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st


# In[ ]:


df=pd.read_csv("creditcard_small.csv")


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


#missing values
df.isnull().sum()


# In[ ]:


df["Class"].value_counts()


# In[ ]:


df['Class'].value_counts(normalize=True)


# In[ ]:


x=df.drop("Class",axis=1)


# In[ ]:


x.isnull().sum()


# In[ ]:


x.fillna(x.mean(),inplace=True)


# In[ ]:


x.isnull().sum()


# In[ ]:


y=df['Class']


# In[ ]:


y.isnull().sum()


# As identified earlier, the `Class` column has one missing value. We need to handle this by dropping the corresponding row from the DataFrame before proceeding with model training.

# In[ ]:


# Drop rows where 'Class' is NaN
df_cleaned = df.dropna(subset=['Class'])

# Re-assign x and y from the cleaned DataFrame
x = df_cleaned.drop("Class", axis=1)
y = df_cleaned['Class']

# Confirm no more missing values in y
print("Missing values in 'Class' after dropping NaN rows:")
st.write(y.isnull().sum())


# Now that the `Class` column is clean, we can re-perform the train-test split and verify that `y_train` no longer contains `NaN` values.

# In[ ]:


# Re-perform train-test split with the cleaned data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Check unique values in y_train again
print("Unique values in y_train after cleaning and re-split:")
st.write(y_train.unique())


# In[ ]:


# Re-perform train-test split (already done in 22ce580a, ensuring x and y are from df_cleaned)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert scaled arrays back to DataFrames with original column names
x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)


# In[ ]:


y_train.unique()


# In[ ]:


model=LogisticRegression(max_iter=5000)


# In[ ]:


y_train = y_train.astype(int)
model.fit(x_train,y_train)


# In[ ]:


plt.hist(x,bins=20)
plt.show()


# In[ ]:


y_pred=model.predict(x_train)


# In[ ]:


accuracy_score(y_train,y_pred)


# In[ ]:


recall_score(y_train,y_pred)


# In[ ]:


confusion_matrix(y_train,y_pred)


# In[ ]:


precision_score(y_train,y_pred)


# In[ ]:


f1_score(y_train,y_pred)


# In[ ]:


model1=DecisionTreeClassifier()


# In[ ]:


model1.fit(x_train,y_train)


# In[ ]:


plt.figure(figsize=(18, 12))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.show()


# In[ ]:


df['Class'].value_counts()


# In[ ]:





# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(x,y)


# In[ ]:


y_resampled.value_counts()


# In[ ]:


plt.figure(figsize=(6, 4))

sns.countplot(x=y_resampled)

plt.title("Balanced Dataset After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()


# In[ ]:


get_ipython().system('pip install shap')


# In[ ]:


import shap


# In[ ]:


prediction = model.predict(x_test.iloc[[0]])

prediction


# In[ ]:


import joblib


# In[ ]:


joblib.dump(model, "fraud_model.pkl")


# In[ ]:


model = joblib.load("fraud_model.pkl")


# In[ ]:


import joblib
import pandas as pd

# Load trained model
model = joblib.load("fraud_model.pkl")

# Example transaction - MUST include all 30 features the model was trained on
# The feature names should also match the training data (x.columns)
# Using dummy values for demonstration, replace with actual transaction data
transaction_data = {
    'Time': [0.0],
    'V1': [-1.35],
    'V2': [0.26],
    'V3': [0.0],
    'V4': [0.0],
    'V5': [0.0],
    'V6': [0.0],
    'V7': [0.0],
    'V8': [0.0],
    'V9': [0.0],
    'V10': [0.0],
    'V11': [0.0],
    'V12': [0.0],
    'V13': [0.0],
    'V14': [0.0],
    'V15': [0.0],
    'V16': [0.0],
    'V17': [0.0],
    'V18': [0.0],
    'V19': [0.0],
    'V20': [0.0],
    'V21': [0.0],
    'V22': [0.0],
    'V23': [0.0],
    'V24': [0.0],
    'V25': [0.0],
    'V26': [0.0],
    'V27': [0.0],
    'V28': [0.0],
    'Amount': [0.0]
}

transaction = pd.DataFrame(transaction_data)

# Now, the 'transaction' data MUST be scaled using the same scaler object that was used for x_train and x_test.
# Since the scaler object is defined in an earlier cell (QMoV2lD2Lbxa) and not globally accessible here,
# a robust solution would involve saving the scaler along with the model.
# For the purpose of immediate execution and resolving the warning, I will modify this cell to include scaling
# of the input 'transaction' data, assuming 'scaler' is still in the kernel state.
# If 'scaler' is not available, then the workflow of saving/loading the scaler will be necessary.

# To proceed, I will assume `scaler` is available globally in the kernel and scale the transaction.
# If this causes an error, it means `scaler` is not global and needs to be saved/loaded.

transaction_scaled = scaler.transform(transaction)
prediction = model.predict(transaction_scaled)

if prediction[0] == 1:
    print("Fraud Transaction")
else:
    print("Non-Fraud Transaction")


# In[ ]:


get_ipython().system('pip install streamlit')


# In[ ]:




st.title("Bank Fraud Detection")

st.write("Welcome to the Fraud Detection System")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Check Transaction"):
    st.write("Transaction details received!")
    st.write("Amount:", amount)


# In[ ]:


get_ipython().system('pip install -q streamlit pyngrok')


# In[ ]:


from pyngrok import ngrok

ngrok.set_auth_token("3HufWZzWvwCPOXbGnhIe5QuJNyC_Eg2VPwxKGmQKHsdYeL4i")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', '\nimport streamlit as st\n\nst.title("Bank Fraud Detection")\n\nst.write("Welcome to the Fraud Detection App")\n\namount = st.number_input("Transaction Amount")\n\nif st.button("Check Transaction"):\n    st.write("Transaction checked!")\n    st.write("Amount:", amount)\n\n\nif amount > 100000:\n    st.error("Suspicious Transaction")\nelse:\n    st.success("Transaction looks normal")\n')


# In[ ]:


get_ipython().system('streamlit run app.py &>/content/logs.txt &')


# In[1]:


public_url = ngrok.connect(8501)

print(public_url)


# In[4]:


import glob
import os

notebook_files = glob.glob("/content/*.ipynb")

if notebook_files:
    notebook = notebook_files[0]
    get_ipython().system('jupyter nbconvert --to python "$notebook"')
else:
    print("Error: No .ipynb files found in /content/. Please save your notebook or specify a path.")


# In[5]:


from google.colab import drive
drive.mount('/content/drive')


# In[6]:


get_ipython().system('find /content/drive/MyDrive -name "*.ipynb" | grep -i "bankfraud"')


# In[7]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Bankfrauddetection.ipynb"')

