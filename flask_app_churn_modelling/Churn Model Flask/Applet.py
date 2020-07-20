from flask import Flask, redirect, request, render_template
app = Flask(__name__)
from keras.models import load_model
import numpy as np
global model,graph
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()

model = load_model("Churn.h5",compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/login", methods = ['POST'])
def login():
    age = request.form['age']
    
    g = request.form['g']
    if (g == 'Female'):
        g = 0
    if (g == 'Male'):
        g = 1
    
    l = request.form['l']
    if (l == 'France'):
        l1, l2 = 0, 0
    if (l == 'Germany'):
        l1, l2 = 1, 0
    if (l == 'Spain'):
        l1, l2 = 0, 1
        
    cc = request.form['cc']
    if (cc == 'Has_cc'):
        cc = 1
    if (cc == 'No_cc'):
        cc = 0
    
    cs = request.form['C_score']        
        
    tenure = request.form['tenure']
    products = request.form['products']
    salary = request.form['salary']
    balance = request.form['balance']
    
    a = request.form['a']
    if (a == 'active'):
        a = 1
    if (a == 'Inactive'):
        a = 0
        
    info = [[l1, l2, cs, g, age, tenure, balance, products, cc, a, salary]]
    with graph.as_default():
        ypred = model.predict(np.array(info))
        y = ypred[0][0]
        if (y == False):
            y = "Will not exit"
        else:
            y = "Will exit"
        
    
    
    return render_template('index.html', abc = y)

if __name__ == '__main__':
    app.run(debug = 'True')