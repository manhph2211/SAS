from flask import Flask, render_template,request
import os
import sys
sys.path.append('../main/')
from local_test import predict


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET','POST'])
def pred():
    
    if request.method == "POST":       
        #get form data
        text = request.form.get('text')
        try:
        	predict = predict(text)  
        except:
        	return "wait wat?"
        return render_template('predict.html',predict= predict)
    pass
    

if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4467)))