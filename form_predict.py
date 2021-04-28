import pickle
from model import CUTOFF
from numpy import isclose

model = pickle.load(open('model.pkl', 'rb'))
choices=['Olark Chat', 'Reference', 'College Website', 'Other']
source_map = {
    choices[0] : [1, 0, 0],
    choices[1] : [0, 1, 0],
    choices[2] : [0, 0, 1],
    choices[3] : [0, 0, 0]
}

def form_predict(data):
    """
    INPUT: list(float, float, string, bool)
    
    RETURNS: {'Yes','No'} 
    """
    X = [
         1.0,                       # statsmodel requires const
         data[0],                   # Number of Visits
         data[1] * 60,              # Seconds spent on Website
         *source_map.get(data[2]),  # Source of the lead
         int(data[3])               # Is user a working professional
    ]
    output = model.predict(X)

    if output >= CUTOFF: return True
    else: return False

