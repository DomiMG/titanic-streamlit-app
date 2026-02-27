###
# Code originally written by Harry Wang (https://github.com/harrywang/mini-ml/)
# It was modified for the purpose of teaching how to deploy a machine learning 
# model using Streamlit.
###

import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import joblib
from datetime import date
import os
from flask import Flask, redirect, url_for, session
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Use a secure random key in production
oauth = OAuth(app)

oauth.register(
  name='oidc',
  authority='https://cognito-idp.us-east-1.amazonaws.com/us-east-1_69xJCrwQX',
  client_id='1pn4koisiu4tvggjhdnkfe3amm',
  client_secret='<client secret>',
  server_metadata_url='https://cognito-idp.us-east-1.amazonaws.com/us-east-1_69xJCrwQX/.well-known/openid-configuration',
  client_kwargs={'scope': 'email openid phone'}
)

@app.route('/')
def index():
    user = session.get('user')
    if user:
        return  f'Hello, {user["email"]}. <a href="/logout">Logout</a>'
    else:
        return f'Welcome! Please <a href="/login">Login</a>.'
    
@app.route('/login')
def login():
    # Alternate option to redirect to /authorize
    # redirect_uri = url_for('authorize', _external=True)
    # return oauth.oidc.authorize_redirect(redirect_uri)
    return oauth.oidc.authorize_redirect('https://d84l1y8p4kdic.cloudfront.net')

@app.route('/authorize')
def authorize():
    token = oauth.oidc.authorize_access_token()
    user = token['userinfo']
    session['user'] = user
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

# load your machine learning model
tree_clf = joblib.load('model_dt.pickle')

# save last modification time of model
model_modification_time = os.path.getmtime('model_dt.pickle')
readable_modification_time = date.fromtimestamp(model_modification_time)

### Streamnlit app code starts here

st.title('Titanic Survival Prediction')
with st.expander('Show sample of Titanic data'):
    df = pd.read_csv('titanic.csv')
    st.dataframe(df.head(20))

# get inputs
with st.sidebar:
    with st.form('inputs'):
        st.markdown('**Please provide passenger information:**')
        sex = st.selectbox('Sex', ['female', 'male'])
        date_of_birth = st.date_input('Date Of Birth', value=date(1888,4,15), min_value=date(1832,1,1), max_value=date(1911,11,13), format="DD/MM/YYYY") # default to modal value
        sib_sp = int(st.number_input('Number of siblings or spouses aboard:', min_value=0, max_value=8, value=0)) # default to modal value
        pclass = st.selectbox('Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
        fare = int(st.number_input('Fare (assume GBP(£)):', min_value=0, max_value=513, value=8)) # default to rounded modal value
        submitted = st.form_submit_button('Predict')

age = date(1912,4,15).year-date_of_birth.year

# this is how to dynamically change text
prediction_state = st.markdown('calculating...')

### Now the inference part starts here

passenger = pd.DataFrame(
    {
        'Pclass': [pclass],
        'Sex': [sex], 
        'Age': [age],
        'SibSp': [sib_sp],
        'Fare': [fare]
    }
)

y_pred = tree_clf.predict(passenger)

proba = tree_clf.predict_proba(passenger)

# Preparing the message to be displayed based on the prediction
if y_pred[0] == 0:
    msg = 'This passenger is predicted to have **died**.'
else:
    msg = 'This passenger is predicted to have **survived**.'

### Now add the prediction result to the Streamlit app

prediction_state.markdown(msg)

st.markdown(f'The survival probability: **{proba[0][1]:.2f}**')
st.title('About the model')
st.markdown('A quarter of available data was used for training.')
st.markdown(f'Model last trained {readable_modification_time}.')
st.markdown('## Limitations')
st.markdown('Age calculated from date of birth is difference in years.')
st.markdown('Inputs are restricted to realistic values.')
