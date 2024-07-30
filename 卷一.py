# @Time: 2024/3/18下午 01:34
# @Author: 余胜辉
# @Email: 3388186690@qq.com
# @File: 2201A-余胜辉-深度一 卷一
# @Software: PyCharm
# 一、	Deep Neural Network:
# 1．导入tensorflow框架(2分)
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
# 2．设置随机种子(2分)
tf.set_random_seed(888)
# 3．使用numpy中的loadtxt读取z7数据集数据(2分)
data = np.loadtxt('z7.csv',delimiter=',')
# 4．使用切片技术提取特征与标签数据(2分)
x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data.shape)
print(y_data.shape)
y_one = len(set(y_data))
print(y_one)
# 5．切分训练集测试集，切分比列自定义(2分)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)
# 6．自定义小批量训练函数(2分)
def next_batch(batch_size):
    global point
    batch_x = x_train[point:point+batch_size]
    batch_y = y_train[point:point+batch_size]
    point += batch_size
    return batch_x,batch_y
# 7．设置特征数据X的占位符(2分)
X = tf.placeholder(tf.float32,shape=[None,16])
# 8．设置标签数据Y的占位符 (2分)
Y = tf.placeholder(tf.int32,shape=[None,])
# 9．调用tf.one_hot处理(6)中数据 (2分)
y_one_hot = tf.one_hot(Y,y_one)
print(y_one_hot)
# 10．设置第一层参数w1,设置参数b1(2分)
w1 = tf.Variable(tf.random_normal(shape=[16,256]))
b1 = tf.Variable(tf.random_normal(shape=[256]))
h1 = tf.sigmoid(tf.matmul(X,w1)+b1)
print(h1)
# 11．设置第二层参数w2,设置参数b2(2分)
w2 = tf.Variable(tf.random_normal(shape=[256,y_one]))
b2 = tf.Variable(tf.random_normal(shape=[y_one]))
# 12．定义预测函数h2(2分)
h2 = tf.matmul(h1,w2)+b2
print(h2)
# 13．设置损失函数 loss(2分)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2,labels=y_one_hot))
print(loss)
# 14．设置优化器(2分)
op = tf.train.AdamOptimizer(0.01).minimize(loss)
print(op)
# 15．定义准确率(2分)
y_pred = tf.argmax(h2,-1)
y_true = y_test
acc = tf.reduce_mean(tf.cast(tf.equal(y_pred,y_true),tf.float32))
print(acc)
# 16．建立会话并调用初始化函数 (2分)
# 17．建立损失值列表(2分)
# 18．设置大循环5次(2分)
# 19．设置批次大小为100(2分)
with tf.Session() as sses:
    sses.run(tf.global_variables_initializer())
    enlhrt = 5
    batch_size = 100
    # 20．计算算总批次数batch(2分)
    # 21．调用小批量函数循环读取批次数据(2分)
    # 22．将每批次损失值添加到列表中(2分)
    # 23．输出损失值，并画出损失图像(2分)
    # 24．计算测试集准确率(2分)
    for i in range(enlhrt):
        point = len(y_train) // batch_size
        for j in range(batch_size):
            batch_x,batch_y = next_batch(batch_size)
            loss_,op_ = sses.run([loss,op],feed_dict={X:batch_x,Y:batch_y})
        print(i,loss_)
    print(sses.run(acc,feed_dict={X:x_test,Y:y_test}))


