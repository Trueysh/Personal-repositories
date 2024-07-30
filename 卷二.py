# @Time: 2024/3/18下午 01:35
# @Author: 余胜辉
# @Email: 3388186690@qq.com
# @File: 2201A-余胜辉-深度一 卷二
# @Software: PyCharm
# 二、	卷积Network(数据在MNIST_data):
# 使用卷积算法实现(数据集MNIST_data):
# (1)	加载Tensorflow库(2分)
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tensorflow.examples.tutorials.mnist import input_data
# (2)	设置随机种子(2分)
tf.set_random_seed(888)
# (3)	调用input_data包中的read_data_sets读取手写数字识别data(2分)
data = input_data.read_data_sets('MNIST_data')
# (4)	提取特征值并利用numpy输出数据维度(2分)
x_data = data.train.images
print(np.array(x_data).shape)
# (5)	提取标签数据并利用numpy输出数据维度(2分)
y_data = data.train.labels
print(np.array(y_data).shape)
y_one = len(set(y_data))
# (6)	创建占位符x(2分)
X = tf.placeholder(tf.float32,shape=[None,784])
# (7)	将X变为满足卷积网络要求的维度(2分)
print(784**0.5)
X_img = tf.reshape(X,shape=[-1,28,28,3])
# (8)	设置y占位符，并对其做tf.one_hot处理(2分)
Y = tf.placeholder(tf.int64,shape=[None,])
y_one_hot = tf.one_hot(Y,y_one)
# (9)	第一层卷积计算，卷积核大小3*3， 步长为1(2分)
w1 = tf.Variable(tf.random_normal(shape=[3,3,3,28]))
c1 = tf.nn.conv2d(X_img,w1,strides=[1,1],padding='SAME')
# (10)	将卷积的结果进行relu激活。(2分)
c1 = tf.nn.relu(c1)
# (11)	进行池化处理，池化核尺寸2*2，步长为2(2分)
c1 = tf.nn.max_pool(c1,ksize=[2,2],strides=[2,2],padding='SAME')
# (12)	第二层卷积计算，卷积核大小3*3， 步长为1(2分)
w2 = tf.Variable(tf.random_normal(shape=[3,3,28,56]))
c2 = tf.nn.conv2d(c1,w2,strides=[1,1],padding='SAME')
# (13)	将卷积的结果进行relu激活。(2分)
c2 = tf.nn.relu(c2)
# (14)	进行池化处理，池化核尺寸2*2，步长为2，(2分)
c2 = tf.nn.max_pool(c2,ksize=[2,2],strides=[2,2],padding='SAME')
# (15)	利用tf.reshape对二层卷积池化后的结果展平(2分)
c2 = c2.shape
print(c2)
dim = 7 * 7 * 56
c2 = tf.reshape(X,[-1,dim])
print(c2)
# (16)	定义全链接层的W和b，并定义预测函数h。(2分)
w = tf.Variable(tf.random_normal(shape=[dim,y_one]))
b = tf.Variable(tf.random_normal(shape=[y_one]))
h = tf.matmul(c2,w)+b
print(h)
# (17)	定义损失函数loss(2分)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h,labels=y_one_hot))
print(loss)
# (18)	设置优化器(2分)
op = tf.train.AdamOptimizer(0.001).minimize(loss)
# (19)	计算准确率。(2分)
y_pred = tf.argmax(h,-1)
y_true = y_data
acc = tf.reduce_mean(tf.cast(tf.equal(y_pred,y_true),tf.float32))
print(acc)
# (20)	建立session，并进行全局变量初始化(2分)
# (21)	设置大循环10次(2分)
# (22)	设置批次大小为100（batch_size=100）(2分)
with tf.Session() as sses:
    sses.run(tf.global_variables_initializer())
    elhret = 10
    batch_size = 100
    # (23)	计算总批次数batch(2分)
    # (24)	调用mnist数据集自带的next_batch循环读取批次数据(2分)
    # (25)	输出每批次损失值。(2分)
    # (26)	计算测试集精度。(2分)
    for i in range(elhret):
        point = len(y_data) // batch_size
        for j in range(batch_size):
            batch_x,batch_y = input_data.read_data_sets('MNIST_data').train.next_batch(batch_size)
            loss_, op_ = sses.run([loss, op], feed_dict={X: batch_x, Y: batch_y})
            print(i, loss_)
        print(sses.run(acc, feed_dict={X:data.test.images, Y:data.test.labels}))
