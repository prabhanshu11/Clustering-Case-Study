import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
#import statsmodels.api as sm

FEATURES = ['TotalVisits', 'Total Time Spent on Website', 'Lead Source_Olark Chat',
            'Lead Source_Reference', 'Lead Source_Welingak Website',
            'What is your current occupation_Working Professional']
CUTOFF = 0.3

#importing data
df = pd.read_csv('Leads.csv')
df = df.replace('Select', np.nan)

to_keep = ['Lead Origin', 'Lead Source', 'Converted', 'TotalVisits',
           'Total Time Spent on Website', 'Page Views Per Visit', 'Country',
           'What is your current occupation', 'A free copy of Mastering The Interview']
df_filtered = df.filter(to_keep, axis='columns')
df_filtered['Lead Origin'] = df_filtered['Lead Origin'].replace(to_replace={'Quick Add Form':'Lead Add Form',
                                                          'Lead Import': 'Lead Add Form'})

# Imputing missing
imp_cat = SimpleImputer(strategy='most_frequent')
imp_cont = SimpleImputer(strategy='median') # median because the distribution was very skewed
cat = ['Lead Source', 'Country', 'What is your current occupation']
cont = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
df_filtered[cat] = imp_cat.fit_transform(df_filtered[cat])
df_filtered[cont] = imp_cont.fit_transform(df_filtered[cont])

# Encoding
enc = LabelEncoder()
col = 'A free copy of Mastering The Interview'
df_filtered[col] = enc.fit_transform(df_filtered[col].values)
cols = ['Lead Origin', 'Lead Source', 'Country', 'What is your current occupation']
df_dummy = pd.get_dummies(data=df_filtered, columns=cols, drop_first=True)


# Splitting y and X
target = 'Converted'
## final_cols are derived from final_model in main.ipynb

X = df_dummy.drop([target], axis=1)[FEATURES]
y = df_dummy[target]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


#Modelling
model = LogisticRegression(max_iter=500)
result = model.fit(X, y)


def make_train_pred(model_result, X, y_actual, cutoff=0.5):
    train_pred = pd.DataFrame(
            {'pred_prob': model_result.predict_proba(X)[:,1],
             'y_actual': y_actual})
    train_pred.insert(column='y_pred', loc=1,
        value=train_pred.pred_prob.map(lambda x: 1 if x>=cutoff else 0))
    return train_pred

train_pred = make_train_pred(result, X, y, cutoff=CUTOFF)

def get_acc_recall(train_pred: 'pd.DataFrame'):
    """
    DataFrame with columns: ['pred_prob', 'y_pred', 'y_actual']
    Returns: accuracy, sensitivity/recall
    """
    from sklearn import metrics
    a = metrics.accuracy_score(train_pred.y_actual, train_pred.y_pred)
    s = metrics.recall_score(train_pred.y_actual, train_pred.y_pred)
    f1 = metrics.f1_score(train_pred.y_actual, train_pred.y_pred)
    return {'Accuracy': np.round(a,2),
            'Sensitivity/Recall' : np.round(s,2),
            'f1 score' : np.round(f1,2)}


# Saving the model
pickle.dump(result, open('model.pkl', 'wb'))

# If this file is run and not imported, print the metrics
if __name__ == '__main__':
    print(get_acc_recall(train_pred))

    
# const                                                     1.0
# TotalVisits                                               5.0
# Total Time Spent on Website                             674.0
# Lead Source_Olark Chat                                    0.0
# Lead Source_Reference                                     0.0
# Lead Source_Welingak Website                              0.0
# What is your current occupation_Working Professional      0.0
