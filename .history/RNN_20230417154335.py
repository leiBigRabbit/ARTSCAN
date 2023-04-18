#用python实现RNN
#用RNN学习二进制加法：1.学习当前位的加法；2.学习关于前一位的进位
import copy,numpy as np
np.random.seed(0)
 
def sigmoid(inX):
       return 1/(1+np.exp(-inX))
 
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
int2binary={}#字典格式
binary_dim=8#二进制最多8位-即2^8
#二进制和十进制的对应
largest_number=pow(2,binary_dim )
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
#unpackbits函数可以把整数转化成2进制数
for i in range(largest_number ):
    int2binary[i]=binary[i]
 
#对网络进行初始化
alpha=0.1
input_dim=2#两个数相加
hidden_dim=16
output_dim=1
#权重值的初始化操作
synapse_0=2*np.random.random((input_dim, hidden_dim))-1
synapse_1=2*np.random.random((hidden_dim, output_dim))-1
synapse_h=2*np.random.random((hidden_dim, hidden_dim))-1
#反向传播更新的参数保存在这里
synapse_0_update=np.zeros_like(synapse_0)
synapse_1_update=np.zeros_like(synapse_1)
synapse_h_update=np.zeros_like(synapse_h)
 
 
#10000次迭代
for j in range(10000):
    #a+b=c
    a_int=np.random.randint(largest_number/2)
    a=int2binary[a_int]
    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]
    c_int=a_int+b_int
    c=int2binary[c_int]#是label值
    d=np.zeros_like(c)#保存预测值
 
    overallError = 0#保存损失值
 
    layer_2_deltas=list()
    layer_1_values=list()#上一个阶段的值
    layer_1_values.append(np.zeros(hidden_dim))
 
    for position in range(binary_dim ):
        x=np.array([[a[binary_dim -position-1],b[binary_dim -position-1]]])
        y=np.array([[c[binary_dim -position-1]]]).T
 
        layer_1=sigmoid(np.dot(x,synapse_0)+np.dot(layer_1_values[-1],synapse_h))#注意点
        #list-表示列表list[-1]表示从右侧开始读取的第一个元素
        
        #dot()返回的是两个数组的点积(dot product)
        #如果处理的是一维数组，则得到的是两数组的內积
        #如果是二维数组（矩阵）之间的运算，则得到的是矩阵积
        layer_2=sigmoid(np.dot(layer_1,synapse_1 ))
        layer_2_error=y-layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))#注意点
        overallError += np.abs(layer_2_error[0])
        d[binary_dim-position-1]=np.round(layer_2[0][0])
        #函数原型是：round(flt, ndig=0)  其中 ndig 是小数点的后面几位(默认为0)，然后对原浮点数 进行四舍五入的操作。
        layer_1_values.append(copy.deepcopy(layer_1))
 
    future_layer_1_delta=np.zeros(hidden_dim)#循环结构传下来的
 
    for position in range(binary_dim):
        x=np.array([[a[position],b[position]]])
        layer_1=layer_1_values[-position-1]
        prev_layer_1=layer_1_values[-position-2]
 
        layer_2_delta=layer_2_deltas[-position-1]
        layer_1_delta=(future_layer_1_delta.dot(synapse_h.T)+layer_2_delta .dot(synapse_1 .T))*sigmoid_output_to_derivative(layer_1)
        #注意点
        #参数更新
        synapse_1_update +=np.atleast_2d(layer_1).T.dot(layer_2_delta)
        #维度改变 :atleast_xd 支持将输入数据直接视为 x维。这里的 x 可以表示：1，2，3。
        synapse_h_update +=np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update +=x.T.dot(layer_1_delta)
 
        future_layer_1_delta =layer_1_delta
 
    synapse_0 +=synapse_0_update *alpha
    synapse_1 +=synapse_1_update *alpha
    synapse_h +=synapse_h_update *alpha
 
    synapse_0_update *=0
    synapse_1_update *=0
    synapse_h_update *=0
 
    if(j % 1000 == 0):
        print("Error:", str(overallError))
        print("Pred:", str(d))
        print("True", str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int)+"+ " +  str(b_int)+"="+str(out))
        print("-------------------------------------")
 
