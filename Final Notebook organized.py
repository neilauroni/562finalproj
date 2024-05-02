#!/usr/bin/env python
# coding: utf-8

# # COMP 562 Final Project Code

# ## Necessary Imports

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import warnings


# In[10]:


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ### Importing the Data

# In[3]:


df = pd.read_csv(".../glioma+grading+clinical+and+mutation+features+dataset (1)/TCGA_InfoWithGrade.csv")


# ### Data Preprocessing

# In[4]:


# X is feature matrix, y is labels
X = df.drop("Grade", axis = 1)
y = df['Grade']


# In[5]:


X


# In[6]:


y


# Applying StandardScaler to Age_at_diagnosis column

# StandardScaler() applies $$ z = \frac{x - \mu}{\sigma} $$ to the Age_at_diagnosis column

# Where z is the standardized value, x is an individual data point, $$ \mu $$ is the mean of the feature, and $$\sigma$$ is the standard deviation of the feature.

# In[7]:


scaler = StandardScaler()
X_scaled = X.copy()
X_scaled['Age_at_diagnosis'] = scaler.fit_transform(X[['Age_at_diagnosis']])


# In[8]:


X_scaled


# ## Correlation Matrix

# In[68]:


X.corr().abs()


# In[72]:


plt.matshow(X.corr().abs())


# In[71]:


sns.heatmap(X.corr().abs(), vmin = 0.0, vmax = 1.0)


# ## Performing Logistic Regression w/ L1 Penality for Variable Selection 

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[29]:


model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)


# In[30]:


logreg.fit(X_train, y_train)


# ### Testing our model

# In[31]:


y_pred = logreg.predict(X_test)


# In[32]:


cross_val = cross_val_score(logreg, X_scaled, y, cv = 10)


# In[33]:


print(f"Cross Validation Performance {cross_val.mean()}")


# In[36]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

if len(logreg.classes_) == 2:
    roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
    print(f"\nROC-AUC: {roc_auc}")


# In[37]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# How do the Coefficents affect the result?

# In[41]:


coefficients = logreg.coef_[0]

coef_df = pd.DataFrame({'Feature': X_scaled.columns, 'Coefficient': coefficients})

coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()


# ## PCA

# In[56]:


pca = PCA(n_components = 2)


# In[57]:


pc = pca.fit_transform(X_scaled)


# ### Plotting principal components

# In[61]:


plt.figure(figsize=(8, 6))
plt.scatter(pc[:, 0], pc[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - First two principal components')
plt.show()


# When n = 3

# In[64]:


pca = PCA(n_components = 3)

pc = pca.fit_transform(X_scaled)


# In[66]:


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.title('PCA - First three principal components')
plt.show()


# ## Ridge

# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)

ridge_cv.fit(X_train, y_train)

best_alpha = ridge_cv.alpha_
print("Best alpha:", best_alpha)

y_pred = ridge_cv.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print("Coefficients:", ridge_cv.coef_)

print("Intercept:", ridge_cv.intercept_)


# In[76]:


lin_coef_df = pd.DataFrame([pd.Series(ridge_cv.coef_), pd.Series(lasso.coef_)])

lin_coef_df

plt.figure(figsize=(10, 5))
ax = sns.heatmap(lin_coef_df, cmap='coolwarm', annot=True, fmt=".2f",
                 annot_kws={"ha": 'center', "va": 'center', "rotation": 45})
plt.title("LASSO and Ridge Coeff")
plt.xlabel("Features")
plt.ylabel("LASSO / Ridge")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()


# ## KNN

# In[134]:


param_grid = {'n_neighbors': range(1, 31)}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=10)  

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[135]:


grid_search.fit(X_train, y_train)


# In[136]:


print("Best n_neighbors:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

best_knn = grid_search.best_estimator_
print("Test set score:", best_knn.score(X_test, y_test))


# In[137]:


roc_auc = roc_auc_score(y_test, best_knn.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC: {roc_auc}")


# In[85]:


best_knn.fit(X_train, y_train)


# In[88]:


y_pred = best_knn.predict(X_test)


# In[89]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# ## Random Forest

# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)


# In[49]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[50]:


y_pred = rf.predict(X_test)


# In[52]:


importances = rf.feature_importances_
importances


# In[54]:


feature_names = X_scaled.columns

indices = np.argsort(importances)[::-1]

sorted_feature_names = [feature_names[i] for i in indices]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances in Random Forest")
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), sorted_feature_names, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()


# In[55]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[142]:


cross_val = cross_val_score(rf, X_scaled, y, cv = 10)
print(f"Cross Validation Performance {cross_val.mean()}")


# In[143]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

if len(logreg.classes_) == 2:
    roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"\nROC-AUC: {roc_auc}")


# In[99]:


single_tree = rf.estimators_[0]


# In[102]:


plt.figure(figsize=(100,50))
plot_tree(single_tree, filled=True, feature_names=X_train.columns, class_names=['LGG', 'GBM'], rounded=True, proportion=True)
plt.show()


# ## Support Vector Classifier

# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[107]:


param_grid = {
    'C': [0.1, 1, 10, 100],  
    'kernel': ['linear', 'rbf', 'poly'],  
    'gamma': ['scale', 'auto'],  
    'degree': [2, 3, 4]  
}


# In[108]:


svc = SVC()


# In[113]:


grid_search = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy', verbose=2, n_jobs = -1)


# This takes too long, find another method

# In[121]:


warnings.filterwarnings("ignore", category=FutureWarning)


# In[122]:


grid_search.fit(X_train, y_train)


# In[123]:


from sklearn.model_selection import RandomizedSearchCV
randomized_search = RandomizedSearchCV(svc, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
randomized_search.fit(X_train, y_train)


# Fuck both of those, I'm just going to guess

# In[125]:


svc = SVC(kernel = 'linear')


# In[126]:


svc.fit(X_train, y_train)


# In[127]:


y_pred = svc.predict(X_test)


# In[129]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# Trying RBF now

# In[130]:


svc_rbf = SVC(kernel = 'rbf', gamma = 'scale')
svc_rbf.fit(X_train, y_train)
y_pred = svc_rbf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[131]:


# CM for Linear
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# ## Neural Networks

# Creating training and test datasets

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Importing necessary libraries

# In[12]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Starting with a simple network, with two hidden layers w/ 4 neurons and a ReLU activation function, and an output layer with a sigmoid activation function.

# In[13]:


model = tf.keras.Sequential([

                               tf.keras.layers.Dense(4, activation = 'relu'), 

                               tf.keras.layers.Dense(4, activation = 'relu'),

                               tf.keras.layers.Dense(1, activation = 'sigmoid')

])


# Use binary_crossentropy loss function for binary classification.

# In[14]:


model.compile( loss= tf.keras.losses.binary_crossentropy,

                optimizer = tf.keras.optimizers.Adam(lr = 0.01),

                metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 100, verbose = 1)


# Checking model accuracy

# In[15]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f' Model loss on the test set: {loss}')
print(f' Model accuracy on the test set: {100*accuracy}')


# In[18]:


y_pred = model.predict(X_test)


# In[19]:


y_pred


# As the sigmoid activation function returns values between 0 and 1, we set predictions > 0.5 to 1, and < 0.5 to 0.

# In[23]:


y_pred_binary = (y_pred > 0.5).astype("int32")
y_pred_binary


# In[20]:


y_test


# In[24]:


cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:




