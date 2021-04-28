from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SubmitField, RadioField, BooleanField
from form_predict import choices

class RegisterForm(FlaskForm):
    visits = IntegerField(label='Total Visits')
    time_minutes = FloatField(label='Total time spent on course website')
    source = RadioField(u'Source', choices=choices) 
    is_professional = BooleanField('Is Professional')
    submit = SubmitField(label='Predict')
