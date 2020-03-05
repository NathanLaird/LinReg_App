from flask import render_template, request,jsonify
from app import app
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask_wtf import FlaskForm
from wtforms import SelectField
import glob
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import numpy as np


UPLOAD_FOLDER = os.getcwd() + '/Data'
ALLOWED_EXTENSIONS = {'csv'}
TRAIN_TEST_SPLIT = 0


#app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


##Helper function for invalid types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## Computes the ROC Json required to visualize in D3
def compute_roc(probs,actual,N=50):
    threshs = []
    TPR =[]
    FPR =[]
    TNR =[]
    FNR =[]
    TPC =[]
    FPC =[]
    TNC =[]
    FNC =[]
    GRAPH_DATA = []
    list_length = len(probs)
    for thresh in np.linspace(min(probs), max(probs), N, endpoint=True):
        fpc = 0
        tpc = 0
        fnc = 0
        tnc = 0
        
        preds = probs >= thresh
        preds = preds.astype(int)
        for pred, real in zip(preds,actual):
            if pred == real:
                if real == 1:
                    tpc+= 1
                else:
                    tnc+= 1
            else:
                if real == 1:
                    fnc+= 1
                else:
                    fpc+= 1
        TPR.append(tpc/list_length)
        FPR.append(fpc/list_length)
        TNR.append(tnc/list_length)
        FNR.append(fnc/list_length)
        threshs.append(thresh)
        TPC.append(tpc)
        FPC.append(fpc)
        TNC.append(tnc)
        FNC.append(fnc)
        GRAPH_DATA.append({'Threshold': thresh, 'y': tpc/(tpc+fnc), 'x': fpc/(fpc+tnc), 'TPC': tpc, 'FPC': fpc, 'TNC': tnc, 'FNC': fnc})
    return GRAPH_DATA


#Generates probablities from data
def get_probs(df,target):
    X = df.drop(columns=[target])
    y = df[target]
    model = LinearRegression().fit(X,y)
    probs = model.predict(X)
    return probs , y


@app.route('/')
@app.route('/index')
def index():
    
    return render_template('index.html', title='Home')


#Handles uploading files to host machine
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))


    return render_template('uploader.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

class Form(FlaskForm):
    csv_files = (glob.glob(UPLOAD_FOLDER + "/*.csv"))

    file = SelectField('file', choices=[(filename.split('/')[-1],filename.split('/')[-1]) for filename in csv_files])
    column = SelectField('column',choices=[])

@app.route('/selector',methods=['GET','POST'])
def selector():
    form = Form()
    csv_files = (glob.glob(UPLOAD_FOLDER + "/*.csv"))
    form.file.choices = [(filename.split('/')[-1],filename.split('/')[-1]) for filename in csv_files]
    form.column.choices = [(1,'Please Select File'),(2,'Then Select Target Column')]
    if request.method =='POST':
        #return '<h1>file: {}, column: {}'.format(form.file.data, form.column.data)
        
        params = (form.file.data, form.column.data)

        return viz(params)

    return render_template('selector.html',form=form)



@app.route('/column/<file>')
def column(file):

    frame = pd.read_csv(UPLOAD_FOLDER+'/'+file)
    columns = list(frame.columns)
    columnArray = []
    
    for column in columns:
        columnObj = {}
        columnObj['id'] = column
        columnObj['name'] = column
        columnArray.append(columnObj)

    return jsonify({'columns':columnArray})



app.route('/viz/<params>')
def viz(params):
    df = pd.read_csv(UPLOAD_FOLDER+'/'+params[0])
    probs, actuals = get_probs(df,params[1])
    chart_data = compute_roc(probs,actuals)
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("viz.html", data=data)



