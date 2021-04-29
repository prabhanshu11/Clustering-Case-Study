import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from forms import RegisterForm
from form_predict import form_predict, get_metrics

app = Flask(__name__)
app.config['SECRET_KEY'] = 'prabhu'


@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def home_page():
    form = RegisterForm()
    if form.validate_on_submit():
        data = [
            form.visits.data,
            form.time_minutes.data,
            form.source.data,
            form.is_professional.data
        ]
        if form_predict(data): 
            prediction = "The lead may Convert"
        else: prediction = "The lead might not Convert"
        return render_template('index_bootstrap.html',form=form, prediction=prediction)

    return render_template('index_bootstrap.html', form=form)

@app.route('/metrics')
def metrics_page():
    items = get_metrics() #items is a dictionary
    return render_template('metrics.html', items=items)

if __name__ == "__main__":
    app.run(debug=True)
