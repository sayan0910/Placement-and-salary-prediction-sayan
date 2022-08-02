

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score



data = pd.read_csv(r"C:\Users\paulp\OneDrive\Desktop\Final_Aman\Placement_Data.csv")







data=data.dropna()






X=data.drop(columns=['status','salary','sl_no','specialisation','etest_p'])









Y=data['salary']










one=OneHotEncoder()




one.fit(X[['gender','ssc_b','hsc_b','hsc_s','degree_t','workex']])






from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline,Pipeline




column_trans=make_column_transformer((OneHotEncoder(categories=one.categories),['gender','ssc_b','hsc_b','hsc_s','degree_t','workex']),remainder='passthrough')









poly_features=PolynomialFeatures(degree=4)
lin_reg=LinearRegression()




pipe=make_pipeline(column_trans,poly_features,lin_reg)









pipe.fit(X,Y)




y_pred=pipe.predict(X)




import pickle




pickle.dump(pipe,open('Poly.pickle','wb'))




k=pipe.predict(pd.DataFrame([['M',67.00,'Others','91','Others','Commerce',58.00,'Sci&Tech','No',58.80]],columns=['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','mba_p']))


print(k)
