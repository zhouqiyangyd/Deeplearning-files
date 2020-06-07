import pandas as pd
import numpy as np
from tqdm import tqdm
'''user = pd.read_csv('./test/test_user.csv')
#app = pd.read_csv('./test/test_app.csv')
#sms = pd.read_csv('./test/test_sms.csv')
voc = pd.read_csv('./test/test_voc.csv')

print(user.info())
print(user['phone_no_m'])
user_cols = user.columns[user.dtypes == 'float64']
user_monthuse = user[user_cols]
user_monthuse = user_monthuse.fillna(0)
user_mean=[]
user['mean_arpu']=user['arpu_202004']
user.pop('county_name')
user.pop('arpu_202004')
user['city_name']=user['city_name'].fillna('未知')
print(user.info())
user.to_csv('./test_user.csv',index=False)


voc.pop('opposite_no_m')
voc.pop('start_datetime')
voc.pop('city_name')
voc.pop('county_name')
voc.pop('imei_m')
print(voc.info())
voc.to_csv('./test_user_voc.csv',index=False)
'''
#整理用户通话频次状态和时间
'''
user=pd.read_csv('./test_user.csv')
print(user.info())
voc=pd.read_csv('./test_user_voc.csv')
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
user_voc.to_csv('test_voc.csv',index=False)'''


#整理用户短信频次状态
'''
user=pd.read_csv('./test_user.csv')
print(user.info())
sms=pd.read_csv('./test/test_sms.csv')
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
user_sms.to_csv('test_sms.csv',index=False)
'''
#整理用户上网频次和花费的流量
'''
user=pd.read_csv('./test_user.csv')
print(user.info())
app=pd.read_csv('./test/test_app.csv')
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
user_app.to_csv('test_app.csv',index=False)
'''

'''
test_user = pd.read_csv('./test_user.csv')
test_voc = pd.read_csv('./test_voc.csv')
test_sms = pd.read_csv('./test_sms.csv')
test_app = pd.read_csv('./test_app.csv')
print(test_user.info())
print(test_voc.info())
print(test_sms.info())
print(test_app.info())
test_data=pd.DataFrame({'phone_no_m':test_user['phone_no_m'].values,'idcard_num':test_user['idcard_cnt'].values,'mean_arpu':test_user['mean_arpu'].values,
                         'call_out_times':test_voc['call_out_times'].values,'cal_in_times':test_voc['cal_in_times'].values,'call_other_times':test_voc['call_other_times'].values,
                         'call_time':test_voc['call_time'].values,'sms_up':test_sms['sms_up'].values,'sms_down':test_sms['sms_down'].values,'app_time':test_app['app_time'].values,
                         'app_flow':test_app['app_flow'].values,'city_name':test_user['city_name'].values})
print(test_data.info())
test_data.to_csv('./test_data.csv',index=False)'''



