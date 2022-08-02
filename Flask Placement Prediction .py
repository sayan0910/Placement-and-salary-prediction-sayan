

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



data = pd.read_csv(r"C:\Users\paulp\OneDrive\Desktop\Final_Aman\Placement_Data.csv")




X=data.drop(columns=['status','salary','sl_no','specialisation','etest_p'])



labelencoder = LabelEncoder()
Y=data['status']
Y = labelencoder.fit_transform(Y)



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)



one=OneHotEncoder()



one.fit(X[['gender','ssc_b','hsc_b','hsc_s','degree_t','workex']])



X['gender']




from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline




column_trans=make_column_transformer((OneHotEncoder(categories=one.categories),['gender','ssc_b','hsc_b','hsc_s','degree_t','workex']),remainder='passthrough')








classifier = KNeighborsClassifier(weights='distance')



pipe=make_pipeline(column_trans,classifier)



pipe.fit(x_train,y_train)



y_pred=pipe.predict(x_test)




accuracy_score(y_test,y_pred)


import pickle

pickle.dump(pipe,open('Knn.pickle','wb'))

k=pipe.predict(pd.DataFrame([['M',67.00,'Others','91','Others','Commerce',58.00,'Sci&Tech','No',58.80]],columns=['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','mba_p']))

print(k)