import cv2
import segmentation
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

def vectorize_image(img_arr):
    r,c = img_arr.shape
    return np.reshape(img_arr,(r*c),order='C')

def train(X_train, y_train, mode=0):
    # 0 = logistic regression
    if(mode == 0):
        model = LogisticRegression(solver='sag',max_iter=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
        
    return model

def evaluate_model(model, X_test, y_test, mode=0):
    # 0 = logistic regression
    if(mode == 0):
        y_pred_lr = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred_lr)
        cr = classification_report(y_test,y_pred_lr)
        
    return cm,cr

X = pd.read_csv('isl_data.csv')
# X = X/255.0
y = pd.read_csv('labels.csv')

X_pca = PCA(n_components=2,whiten=True).fit_transform(X)

X_isomap = Isomap(n_neighbors=10).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

X_train_isomap, X_test_isomap, y_train_isomap, y_test_isomap = train_test_split(X_isomap, y, test_size=0.2, random_state=42)



model = train(X_train, y_train)
cm,cr = evaluate_model(model,X_test,y_test)
print(cm)
print(cr)

model_pca = train(X_train_pca, y_train_pca)
cm,cr = evaluate_model(model_pca,X_test_pca,y_test_pca)
print(cm)
print(cr)

# model_isomap = train(X_train_isomap, y_train_isomap)
# cm,cr = evaluate_model(model_isomap,X_test_isomap,y_test_isomap)
# print(cm)
# print(cr)

# test_img = cv2.imread('test/1.jpg')
# test_img = cv2.resize(test_img,(64,64))
# test_img_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("test",test_img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# X_test_img = vectorize_image(test_img_gray)
# X_test_img = X_test_img/255.0
# X_test_img = X_test_img.reshape((1,4096))
# print(X_test_img)
# y_test_img = np.array(['A'])
# y_test_img = y_test_img.reshape((1,1))
# cm,cr = evaluate_model(model,X_test_img,y_test_img)
# print(cm)
# print(cr)

# log_regr = LogisticRegression(solver='sag',max_iter=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
# y_pred_lr = log_regr.predict(X_test)
# print("Logistic Regression")
# print(confusion_matrix(y_test,y_pred_lr))
# print(classification_report(y_test,y_pred_lr))
# print("\n")

# log_regr = LogisticRegression(solver='sag',max_iter=10, n_jobs=-1, verbose=1).fit(X_train_pca, y_train_pca)
# y_pred_lr = log_regr.predict(X_test_pca)
# print("Logistic Regression")
# print(confusion_matrix(y_test_pca,y_pred_lr))
# print(classification_report(y_test_pca,y_pred_lr))
# print("\n")