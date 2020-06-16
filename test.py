import pandas as pd
import numpy as np
from tqdm import tqdm
user = pd.read_csv('./train/train_user.csv')
#app = pd.read_csv('./train/train_app.csv')
#sms = pd.read_csv('./train/train_sms.csv')
voc = pd.read_csv('./train/train_voc.csv')
'''print(app.info())
print(sms.info())
'''

print(user.info())
months=['arpu_201908','arpu_201909','arpu_201910','arpu_201911','arpu_201912','arpu_202001','arpu_202002','arpu_202003']
#user_cols = user.columns[user.dtypes == 'float64']
k=0
a=[]
user_months=[]
for name in tqdm(user['phone_no_m'].values):
    for month in months:
        if user[user['phone_no_m']==name][month].isna().values==False:
            k=k+1
    user_months.append(k)
    print(k)
    k=0
user['user_months']=user_months

user_monthuse = user[months]
user_monthuse = user_monthuse.fillna(0)
user_mean=[]

for i in tqdm(range(user_monthuse.values.shape[0])):
    user_mean .append(np.sum(user_monthuse.iloc[i,:]))
user['mean_arpu']=user_mean
for i in range(user.values.shape[0]):
    if user.loc[i,'mean_arpu']<=0:
        usera=user.drop(i)

for i in months:
    user.pop(str(i))
label = user.pop('label')
user.pop('county_name')
print(user['city_name'].value_counts())
user['city_name']=user['city_name'].fillna('未知')
print(user)
user.to_csv('./train_user.csv',index=False)

voc.pop('opposite_no_m')
voc.pop('start_datetime')
voc.pop('city_name')
voc.pop('county_name')
voc.pop('imei_m')
usemonths=[]
print(voc.info())
voc.to_csv('./train_voc.csv',index=False)

'''

#整理用户通话频次状态和时间
user=pd.read_csv('./train_user.csv')
print(user.info())
voc=pd.read_csv('./train_voc.csv')
print(voc.info())
print(voc.head())
user_m = user['phone_no_m'].values
print(user_m)
voc_out=[]
voc_in=[]
voc_other=[]
voc_time=[]
for name in tqdm(user_m):
    voc_out.append(voc[(voc['phone_no_m'] == name) & (voc['calltype_id']==1)]['phone_no_m'].count())
    voc_in.append(voc[(voc['phone_no_m'] == name) & (voc['calltype_id'] == 2)]['phone_no_m'].count())
    voc_other.append(voc[(voc['phone_no_m'] == name) & (voc['calltype_id'] == 3)]['phone_no_m'].count())
    voc_time.append(voc[(voc['phone_no_m'] == name)]['call_dur'].sum())
voc_out=np.array(voc_out)
voc_in=np.array(voc_in)
voc_other=np.array(voc_other)
voc_time=np.array(voc_time)
user_voc = pd.DataFrame({'phone_no_m':user_m,'call_out_times':voc_out,'cal_in_times':voc_in,'call_other_times':voc_other,'call_time':voc_time})
user_voc.to_csv('user_voc.csv',index=False)

#整理用户短信频次状态

user=pd.read_csv('./train_user.csv')
print(user.info())
sms=pd.read_csv('./train/train_sms.csv')
print(sms.info())
print(sms.head())
user_m = user['phone_no_m'].values
print(user_m)
sms_up=[]
sms_down=[]
for name in tqdm(user_m):
    sms_up.append(sms[(sms['phone_no_m'] == name) & (sms['calltype_id']==1)]['phone_no_m'].count())
    sms_down.append(sms[(sms['phone_no_m'] == name) & (sms['calltype_id'] == 2)]['phone_no_m'].count())
user_m=np.array(user_m)
sms_up=np.array(sms_up)
sms_down=np.array(sms_down)
user_sms = pd.DataFrame({'phone_no_m':user_m,'sms_up':sms_up,'sms_down':sms_down})
user_sms.to_csv('user_sms.csv',index=False)

#整理用户上网频次和花费的流量

user=pd.read_csv('./train_user.csv')
print(user.info())
app=pd.read_csv('./train/train_app.csv')
print(app.info())
print(app.head())
user_m = user['phone_no_m'].values
print(user_m)
app_time=[]
app_flow=[]
for name in tqdm(user_m):
    app_time.append(app[(app['phone_no_m'] == name)]['phone_no_m'].count())
    app_flow.append(app[(app['phone_no_m'] == name)]['flow'].sum())
user_m=np.array(user_m)
app_time=np.array(app_time)
app_flow=np.array(app_flow)
user_app = pd.DataFrame({'phone_no_m':user_m,'app_time':app_time,'app_flow':app_flow})
user_app.to_csv('user_app.csv',index=False)



train_user = pd.read_csv('./train_user.csv')
train_voc = pd.read_csv('./user_voc.csv')
train_sms = pd.read_csv('./user_sms.csv')
train_app = pd.read_csv('./user_app.csv')
print(train_user.info())
print(train_voc.info())
print(train_sms.info())
print(train_app.info())
train_data=pd.DataFrame({'phone_no_m':train_user['phone_no_m'].values,'idcard_num':train_user['idcard_cnt'].values,'mean_arpu':train_user['mean_arpu'].values,
                         'call_out_times':train_voc['call_out_times'].values,'cal_in_times':train_voc['cal_in_times'].values,'call_other_times':train_voc['call_other_times'].values,
                         'call_time':train_voc['call_time'].values,'sms_up':train_sms['sms_up'].values,'sms_down':train_sms['sms_down'].values,'app_time':train_app['app_time'].values,
                         'app_flow':train_app['app_flow'].values,'city_name':train_user['city_name'].values})
print(train_data.info())
train_data.to_csv('./train_data.csv',index=False)'''



