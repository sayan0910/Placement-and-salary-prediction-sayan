
#from crypt import methods
from flask import app, request
from flask import Flask,render_template,url_for
import pickle
import pandas  as pd

placement=pd.read_csv('Placement_Data.csv')

model_1=pickle.load(open("Knn.pickle","rb"))

model_2=pickle.load(open("Poly.pickle","rb"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ec9439cfc6c796ae2029594d'

@app.route('/')
@app.route('/Home')
def home_page():
    gender=sorted(placement['gender'].unique())
    ssc_b=sorted(placement['ssc_b'].unique())
    hsc_b=sorted(placement['hsc_b'].unique())	
    hsc_s=sorted(placement['hsc_s'].unique())
    degree_t=sorted(placement['degree_t'].unique())	
    workex=sorted(placement['workex'].unique())

   
    return render_template('home.html',gender=gender,ssc_b=ssc_b,hsc_b=hsc_b,hsc_s=hsc_s,degree_t=degree_t,workex=workex)

@app.route('/predict',methods=["post"])

def predict():
    Name=request.form.get('Name')
    gender=request.form.get('gender')
    ssc_b=request.form.get('ssc_b')
    hsc_b=request.form.get('hsc_b')
    hsc_s=request.form.get('hsc_s')
    degree_t=request.form.get('degree_t')
    workex=request.form.get('workex')
    SEP=float(request.form.get('SEP'))
    HEP=float(request.form.get('HEP'))
    DP=float(request.form.get('DP'))
    MBAP=float(request.form.get('MBAP'))


    prediction_placement=model_1.predict(pd.DataFrame([[gender,SEP,ssc_b,HEP,hsc_b,hsc_s,DP,degree_t,workex,MBAP]],columns=['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','mba_p']))
    prediction_salary=model_2.predict(pd.DataFrame([[gender,SEP,ssc_b,HEP,hsc_b,hsc_s,DP,degree_t,workex,MBAP]],columns=['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','mba_p']))

    return render_template("start.html",Name=Name, prediction_salary=prediction_salary, prediction_placement=prediction_placement)





if __name__ =='__main__':
    app.run(debug=True,port=8088)