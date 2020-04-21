import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

index = 25
#plt.imshow(train_set_x_orig[index])
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
"将维度为（64，64，3）的图片数组重构为（64*64*3，1）的数组，转置变为为特征值X1（12288，1）的矩阵" \
"由209个图片组成（12288，209）的训练集矩阵X" \
"同理测试集为（12288，50）的矩阵"
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
"由于颜色为0-255的范围将每个通道的值除以255可以让数据更集中更标准"
####################################################################
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
"参数为z，输出为函数计算值"
#测试print (sigmoid(0.8))


def initialize_with_zeros(dim):
    w=np.random.rand(dim,1)*1e-5
    #w=np.zeros(shape = (dim,1))将w用0填充为dim行1列的矩阵
    "Logistic回归没有隐藏层。 如果将权重初始化为零，则Logistic回归中的第一个示例x将输出零，" \
    "但Logistic回归的导数取决于不是零的输入x（因为没有隐藏层）。" \
    " 因此，在第二次迭代中，如果x不是常量向量，则权值遵循x的分布并且彼此不同。"
    b= 0
    return w,b

def propagate(w,b,X,Y):
    "输入w-特征值，b-实数，X训练集矩阵，Y测试集矩阵，cost逻辑回归的负对数似然成本"
    "dw-w的损失梯度，db-b的损失梯度"
    m=X.shape[1]
    "正向传播"
    A=sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))#计算成本
    "反向传播"
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)

    grads={
        "dw":dw,
        "db":db
    }#创建一个字典保存dw和db
    return (grads,cost)

#使用梯度下降函数优化w和b
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    #输出已学习的w和b的值，我们可以使用w和b来预测数据集X的标签
    #参数
    "num_iterations  - 优化循环的迭代次数"
    "learning_rate  - 梯度下降更新规则的学习率"
    "print_cost  - 每100步打印一次损失值"
    #返回值
    "param 包含w和b的字典，grads 包含dw和db的字典，成本 优化期间的所有成本列表"
    costs = []
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        "更新w和b"
        if i % 50==0:
            costs.append(cost)
        if (print_cost)and(i%50==0):
            print("迭代次数：%i,误差值：%f,学习速率：%f"%(i,cost,learning_rate))

    params={"w":w,"b":b}
    grads={"dw":dw,"db":db}

    return (params,grads,costs)

#预测函数
def predict(w,b,X):
    "参数：w 特征数组，b 偏差实数，X 训练集矩阵"
    "返回值： Y_prediction 包含X中所以图片的所以预测二元分类的数组"
    m = X.shape[1]
    "图片的数量"
    Y_prediction=np.zeros((1,m))
    "用0填充1行m列的矩阵Y_prediction"
    w=w.reshape(X.shape[0],1)

    # 预测猫在图片中出现的概率
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0
        "将预测值进行简单的二元分类>0.5代表"
        return  Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iterations,learning_rate,print_cost):
    """通过调用之前实现的函数来构建逻辑回归模型
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率
        print_cost  - 设置为true以每100次迭代打印成本

    返回：
        d  - 包含有关模型信息的字典。"""
    w,b=initialize_with_zeros(X_train.shape[0])
    print(w, len(w))

    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w ,b = parameters["w"] , parameters["b"]

    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    print("训练集准确性："  , format(1- np.mean(np.abs(Y_prediction_train - Y_train))))
    print("测试集准确性："  , format(1- np.mean(np.abs(Y_prediction_test - Y_test))))
    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d
print("-----------------------------测试-----------------------")
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.01, print_cost = True)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

"""
learning_rates=[0.5,0.1,0.01,0.001]
models={}
for i in learning_rates:
    print ("学习效率："+str(i))
    d=model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000,learning_rate=i, print_cost = True)
    print("--------------------------------------------------------")
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label=str(models[str(i)]["learning_rate"]))

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
"""