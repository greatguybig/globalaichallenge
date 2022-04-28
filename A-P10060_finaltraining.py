import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from lunar_python import Lunar,Solar
import holidays
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')
input_path = ''
model_path = ''
input_csv1 = 'CoolingLoad15months.csv'
input_csv2 = 'July_to_September.csv'


def select(data):
    datalist = []
    i = 100
    print(len(data))
    while i < len(data):
        if data.iloc[i].isnull().any():
            i += 1
            continue
        else:
            for j in range(i+1,len(data)):
                if (pd.isnull(data.iloc[j,:6]).any() and pd.isnull(data.iloc[j+1,:6]).any()):
                    e = j - i
                    print('e',e)
                    if e > 48:
                        select = data.iloc[i:j-1]
                        datalist.append(select)
                        print('select',i)
                        i = j+2                        
                    else:
                        i = j+2
                    break
                if (j == len(data)-1):
                    select = data.iloc[i:j+1]
                    datalist.append(select)
                    print('e',j-i)
                    print('select',i)
                    i = j + 1
                    break
    selection = pd.concat(datalist)
#    selection.to_csv('slection1.csv')
    print(selection.index)
    for i in ['Average_OAT','Humidity','UV_Index','NT_CoolingLoad','ST_CoolingLoad']:
        for j in selection.index:
            mean = np.mean(selection[(selection['jieqi']==selection.loc[j,'jieqi'])&(selection['time'] ==selection.loc[j,'time'])][i])                                     
            deviation = np.std(selection[(selection['jieqi']==selection.loc[j,'jieqi'])&(selection['time'] ==selection.loc[j,'time'])][i])                                     
            if (i in ['Average_OAT','Humidity','UV_Index']):
                if ((abs(selection.loc[j,i] - mean) > 3*deviation) or (pd.isnull(selection.loc[j,i]))):
                    uvmean = np.mean(selection[(selection['jieqi']==selection.loc[j]['jieqi'])
                                  &(selection['time'] ==selection.loc[j,'time'])
                                  &(selection.loc[:,i] != selection.loc[j,i])][i])                                    
                    selection.loc[j,i] = uvmean
                    print('weather outlier',j,i)
            mean = np.mean(selection[(selection['weekday']==selection.loc[j,'weekday'])
                                    &(selection['time'] ==selection.loc[j,'time'])
                                    &(selection['holiday'] ==selection.loc[j,'holiday'])][i])
            deviation = np.std(selection[(selection['weekday']==selection.loc[j,'weekday'])
                                    &(selection['time'] ==selection.loc[j,'time'])
                                    &(selection['holiday'] ==selection.loc[j,'holiday'])][i])
            if (i in ['NT_CoolingLoad','ST_CoolingLoad']):
                if ((abs(selection.loc[j,i] - mean) > 3*deviation) or (selection.loc[j,i]<=0)):
                    loadmean = np.mean(selection[(selection['weekday']==selection.loc[j,'weekday'])
                                    &(selection['time'] ==selection.loc[j,'time'])
                                  &(selection.loc[:,i] != selection.loc[j,i])
                                    &(selection['holiday'] ==selection.loc[j,'holiday'])][i])
                    selection.loc[j,i] = loadmean
                    print('load outlier',j,i)
            selection.loc[j,'CoolingLoad'] = selection.loc[j,'NT_CoolingLoad']+selection.loc[j,'ST_CoolingLoad']
#    selection.to_csv('selection.csv')
    if selection.isnull().any().any():
        print('null'*10)
    selectlist = []
    for i in range(len(datalist)):
        piece = selection.loc[datalist[i].index]
        selectlist.append(piece)
#        piece.to_csv('selection2//select{}.csv'.format(i))
    return selectlist
def param(data):
    raw = data
    timestr = pd.to_datetime(raw.index,errors='coerce')
    raw['date'] = timestr.date
    raw['time'] = timestr.time
    raw['year'] = timestr.year
    raw['day'] = timestr.day
    raw['datetime'] = timestr
    raw['month'] = timestr.month
    raw['hour'] = timestr.hour
    hour = timestr.hour
    raw['minute'] = timestr.minute
    minute = timestr.minute
    raw['second'] = timestr.second
    second = timestr.second
    raw['seconds'] = 3600*hour+60*minute+second
    week = timestr.dayofweek
    raw['weekday'] = week + 1
    raw['jieqi'] = raw['weekday']
    raw['holiday'] = np.zeros(len(raw))
    raw['workinghour'] = np.zeros(len(raw))
    jieqidic = {'立春':1,
                '雨水':2,
                '惊蛰':3,
                '春分':4,
                '清明':5,
                '谷雨':6,
                '立夏':7,
                '小满':8,
                '芒种':9,
                '夏至':10,
                '小暑':11,
                '大暑':12,
                '立秋':13,
                '处暑':14,
                '白露':15,
                '秋分':16,
                '寒露':17,
                '霜降':18,
                '立冬':19,
                '小雪':20,
                '大雪':21,
                '冬至':22,
                '小寒':23,
                '大寒':24
                }
    holiday = holidays.HongKong()
    for i in raw.index:
        date = Lunar.fromDate(raw.loc[i,'datetime'])
        d = Lunar.fromYmd(date.getYear(),date.getMonth(),date.getDay())
        raw.loc[i]['jieqi'] = jieqidic.get(d.getPrevJieQi().getName())
        if raw.loc[i]['date'] in holiday:
            raw.loc[i]['holiday'] = 1
        if ((raw.loc[i,'seconds']>=30600)&(raw.loc[i,'seconds']<64800)):
            raw.loc[i,'workinghour'] = 1
        
    return raw
def preprocess(data,k):
    processedx = []
    processedy = []
    for i in range(k-1,len(data)):
        row = data.iloc[i-k+1:i+1,:-1].to_numpy()
        temprow = row[-1,:]
        for j in range(0,row.shape[0]-1):
            temp2 = row[j,:]            
            temprow = np.concatenate([temprow,temp2])
            
#            print(temprow.shape)
        temprow = list(temprow)
        processedx.append(temprow)
        processedy.append(data.iloc[i,-1])
    x = np.array(processedx)
    y = np.array(processedy)
#    print(x.shape,y.shape)
    return x,y

data = pd.read_csv(input_path+input_csv1,index_col = 0)
data1 = pd.read_csv(input_path+input_csv2,index_col = 0)
data = pd.concat([data,data1],axis = 0)

data = param(data)                                                            
predata = select(data)
for i in range(len(predata)):
    predata[i] = predata[i].loc[:,['seconds','workinghour','Average_OAT','Humidity','UV_Index',
                        'Average_Rainfall','weekday','jieqi','holiday','CoolingLoad']]
testdata = predata.pop(-1)
traindata = predata
er = []
params = {
    'n_estimators':[100,200,300,400,500],
    'objective': ['reg:squarederror', 'reg:tweedie'],
    'booster': ['gbtree'],
    'eval_metric': ['rmse'],
    'eta': [i/100.0 for i in range(1,5)]
    
}
n_iter_search = 200

for k in [8]:
    x = []
    y = []
    for i in range(len(traindata)):                
        trainx,trainy = preprocess(traindata[i],k)
        x.append(trainx)
        y.append(trainy)
        print(trainx.shape,trainy.shape)
    x = np.concatenate(x,axis = 0)
    y = np.concatenate(y)
#    print(x.shape,y.shape)
    reg = XGBRegressor(objective='reg:squarederror')
    random_search = RandomizedSearchCV(reg, param_distributions=params,
                                   n_iter=n_iter_search, cv=5,  scoring='neg_mean_squared_error')
    random_search.fit(x,y)
    best_regressor = random_search.best_estimator_
    best_regressor.save_model(model_path+'Xgbestmodel'+str(k)+'.json')
    testx,testy = preprocess(testdata,k)
    x_test = testx
    y_test = testy
    y_pred=best_regressor.predict(x_test)
    print(random_search.best_params_)
    plt.figure(figsize = (14,5))
    print(x_test.shape,y_pred.shape)
    plt.plot(y_pred,color = 'r',label = 'prediction')
    plt.plot(y_test,color = 'g',label = 'actual')
    plt.legend()
    plt.show()
    rmse = np.sqrt(np.mean(np.square(y_pred-y_test)))
    print('rmse'+str(k),rmse)
    er.append(rmse)
