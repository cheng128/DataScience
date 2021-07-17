import sys
import pandas as pd
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


epoch = sys.argv[1]
lr = sys.argv[2]
train_df = pd.read_csv('nthuds2021hw3/train.csv')
test_df = pd.read_csv('nthuds2021hw3/test.csv')

impute_knn = KNNImputer(n_neighbors=21)

xgbc = XGBClassifier(n_estimators=int(epoch),
                     eta=float(lr),
                     subsample=0.5,
                     colsample_bytree=0.5)

def trans_date(df):
    df.Date = pd.to_datetime(df.Date)
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    del df["Date"]
    return df

def impute_value(df, float_df):
    float_df = df.select_dtypes(include='float')
    value = impute_knn.fit_transform(float_df)

    for idx, col in enumerate(float_df.columns):
        temp = [i[idx] for i in value]
        df[col] = temp

    df['Year'].apply(lambda x: int(x))
    df['Month'].apply(lambda x: int(x))
    df['Day'].apply(lambda x: int(x))
    
    pd.get_dummies(df)
    return df

def over_under_sampling(target, label):
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.4)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    target, label = pipeline.fit_resample(target, label)
    
    return target, label

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=5, 
                                                    train_size=0.9)

    evalset = [(X_train, y_train), (X_test,y_test)]

    trained_model = xgbc.fit(X_train, y_train, 
                             eval_metric='logloss', 
                             eval_set=evalset)

    y_pred = trained_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print('Accuracy: %.3f' % score)
    print('f1-score: ', f1_score(y_pred, y_test))
    print('recall: ', recall_score(y_pred, y_test))
    print('precision: ', precision_score(y_pred, y_test))
    
    return trained_model
    
def main():
    train_df = trans_date(train_df)
    train_df['label'] = train_df['Weather']
    del train_df['Weather']
    
    X, y = train_df[train_df.columns[:-1]], train_df['label']
    X = impute_value(X)

    test_df = trans_date(test_df)
    test_df = impute_value(test_df)

    for col in X.select_dtypes(include=['float']):
        X[col] = preprocessing.scale(X[col])

    X, y = over_under_sampling(X, y)

    trained_model = train_model(X, y)
    test_res = trained_model.predict(test_df)

    index = range(len(test_res))
    df = pd.DataFrame({'Id': index, 'Weather': test_res})

    with open(epoch+"_"+lr+'.csv', 'w') as g:
        g.write(df.to_csv(index=False))
        
if __name__ == '__main__':
    main()