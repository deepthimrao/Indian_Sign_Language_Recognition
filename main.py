import cv2
import segmentation
import os
import sys
import pandas as pd
import data_creation as dc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

dataset_path = 'dataset/'
data = {}

for label in CLASSES:
    binary_images_list = []
    for img in os.listdir(dataset_path+label):
        # print(img)
        image = cv2.imread(dataset_path+label+'/'+img)
        image = cv2.resize(image,(64,64))
        binary_image = segmentation.segment(image)
        binary_images_list.append(binary_image)
        # cv2.namedWindow("test",cv2.WINDOW_NORMAL)
        # cv2.imshow("test",binary_image)
        # print(binary_image)
        # if(cv2.waitKey(0) == ord('a')):
        #     cv2.destroyAllWindows()
        #     sys.exit()
        # break
    data[label] = binary_images_list

# cv2.destroyAllWindows()

# df,df_labels = dc.create_data(data)
df_labels = dc.create_data(data)
# df.to_csv('isl_data.csv',index=False)
# X=df
# y=df_labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# log_regr = LogisticRegression(solver='sag',max_iter=100, n_jobs=-1).fit(X_train, y_train)
# y_pred_lr = log_regr.predict(X_test)
# print("Logistic Regression")
# print(confusion_matrix(y_test,y_pred_lr))
# print(classification_report(y_test,y_pred_lr))
# print("\n")