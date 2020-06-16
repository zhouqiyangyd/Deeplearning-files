import pandas as pd
import  numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from  tensorflow import  keras
from tensorflow.keras import layers,losses,Sequential,optimizers
import time
from tqdm import tqdm

config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess =tf.compat.v1.Session(config=config)
#=======================================================================
def normalization(list):
    nor=np.ones((6106,10))
    for i in range(10):
        nor[:,i] = (list[:,i] - np.mean(list[:,i]))/np.std(list[:,i])+1e-8
    return nor
#=======================================================================
class Baisblock(layers.Layer):
    def __init__(self,filter_num):
        super(Baisblock,self).__init__()
        self.r1 = layers.Dense(filter_num,activation='relu')
        self.r2 = layers.Dense(filter_num)
        self.bn = layers.BatchNormalization()



    def call(self, inputs, training= None):
        x = self.r1(inputs,training=training)
        x = self.r2(x, training=training)
        x = self.bn(x, training=training)
        x = layers.add([x,inputs])
        return x
#=======================================================================
class ResNet (keras.Model):
    def __init__(self,layer_dims,num_classes=100):
        super(ResNet,self).__init__()
        self.getdata = keras.Sequential([
            layers.Dense(128)
        ])
        self.layer1 = self.build_resblok(128,layer_dims[0])
        self.layer2 = self.build_resblok(128, layer_dims[1])
        self.layer3 = self.build_resblok(128, layer_dims[2])
        self.layer4 = self.build_resblok(128, layer_dims[3])
        self.r = layers.Dense(64,activation='relu')
        self.drop2=layers.Dropout(0.5)
        self.out = layers.Dense(1,activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.getdata(inputs,training=training)
        l1 = self.layer1(x,training=training)
        x = self.layer2(l1,training=training)
        l3 = self.layer3(x, training=training)
        x = self.layer4(l3, training=training)
        x= self.r(x,training=training)
        x = self.drop2(x)
        x = self.out(x)

        return x

    def build_resblok(self, filter_num,blocks,stride=1):
        res_block = keras.Sequential()
        res_block.add(Baisblock(filter_num))
        for _ in range(1,blocks):
            res_block.add(Baisblock(filter_num))
        return res_block
#=======================================================================
train_label = pd.read_csv('./train/train_user.csv')['label']
print('检查标签是否有缺失值\n',train_label[train_label.isnull().values==True])
train_label=train_label.values
print('==============================标签整理完毕',train_label.shape,'===============================')

train_data=pd.read_csv('train_data.csv')
user = pd.read_csv('train_user.csv')
user.loc[(user['user_months']==0),'user_months']=1e-8
usemonth =user['user_months']
#先不添加城市
quitmonth=['call_out_times','cal_in_times','call_other_times','call_time','sms_up','sms_down','app_time','app_flow']


#train_data.pop('mean_arpu')
for i in quitmonth:
    train_data[i]=train_data[i].values/usemonth.values
train_data.pop('phone_no_m')
train_data['out/in']=train_data['call_out_times']/(train_data['cal_in_times']+1e-8)
data_type = train_data.columns[train_data.dtypes != 'object']
train_data.loc[:, data_type] = (train_data.loc[:, data_type] - train_data.loc[:, data_type].mean()) / (train_data.loc[:, data_type].std()+1e-8)

'''
train_data['unknowuse']=train_data['mean_arpu']
train_data.loc[(train_data['mean_arpu'].isnull()),'unknowuse']=1
train_data.loc[(train_data['mean_arpu'].notnull()),'unknowuse']=0
print(train_data['unknowuse'])
train_data['mean_arpu']=pd.read_csv('train_user.csv')['mean_arpu'].values/pd.read_csv('train_user.csv')['user_months'].values
train_data['mean_arpu']=train_data['mean_arpu'].fillna(0)
'''


print('检查训练集是否有缺失值\n',train_data[train_data.isnull().values==True])
train_data = pd.get_dummies(train_data)
train_data=train_data.values
print('==============================训练集整理完毕',train_data.shape,'===============================')

#========================================================================
test_data=pd.read_csv('test_data.csv')


#test_data.pop('mean_arpu')



test_data['out/in']=test_data['call_out_times']/(test_data['cal_in_times']+1e-8)
test_data_type = test_data.columns[test_data.dtypes != 'object']
#test_data.loc[:,quitmonth]=test_data.loc[:,quitmonth]/8
test_data.loc[:,test_data_type]=(test_data.loc[:,test_data_type]-test_data.loc[:,test_data_type].mean())/(test_data.loc[:,test_data_type].std()+1e-4)

'''
test_data['unknowuse']=test_data['mean_arpu']
test_data.loc[(test_data['mean_arpu'].isnull()),'unknowuse']=1
test_data.loc[(test_data['mean_arpu'].notnull()),'unknowuse']=0
print(test_data['unknowuse'])

test_data['mean_arpu']=test_data['mean_arpu'].fillna(0)
'''


test_data.pop('phone_no_m')
print('检查测试集是否有缺失值\n',test_data[test_data.isnull().values==True])
test_data=pd.get_dummies(test_data)
test_data=test_data.values
print('==============================测试集整理完毕',test_data.shape,'===============================')




#========================================================================

nets=[[1,2,4,8],[0,0,0,0]]
historys=[]
pres=[]
k=0
model = ResNet([1,2,4,8])
model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.binary_crossentropy,
            metrics = ['accuracy'])
model.build(input_shape=(None,35))
historys=model.fit(train_data,train_label,batch_size=32,epochs=100,validation_split=0.05)
plt.plot(historys.history['val_accuracy'])
plt.plot(historys.history['val_loss'])
plt.legend([str(nets[k])+'val_accuracy',str(nets[k])+'val_loss'])
plt.show()
pre=[]
print(test_data.shape)
pres=model.predict(test_data)
for i in pres:
    print(i)
    if i >0.5:
        pre.append(1)
    if i <=0.5:
        pre.append(0)
name=pd.read_csv('./test_user.csv')['phone_no_m'].values
name = np.array(name)
pre = np.array(pre)
print(name.shape,pre.shape)
ans = pd.DataFrame({'phone_no_m':name,'label':pre})
print(time.time())
ans.to_csv(str(time.time())+'ans.csv',index=False)

