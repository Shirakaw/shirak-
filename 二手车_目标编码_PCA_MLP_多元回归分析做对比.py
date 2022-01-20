#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time
from tqdm import tqdm
import itertools

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA


## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import scipy.signal as signal
train_data = pd.read_csv('new7.csv')


# In[2]:


train_data.head()


# In[3]:


train_data.columns


# In[4]:


train_data.info()


# In[5]:



train_data['brand_carCode_maketype'] = train_data['brand_carCode_maketype'].astype('int')
train_data['serial_model_modelyear'] = train_data['serial_model_modelyear'].astype('int')                


# In[6]:


train_data.info()


# In[5]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders.target_encoder import TargetEncoder
import numpy as np
import pandas as pd
X_encoder = train_data.drop(['carid', 'price'], axis=1)
y_encoder = train_data['carid']
enc = TargetEncoder(cols=[ 'color', 'cityId', 'transferCount',  'country', 'displacement', 'gearbox',
       'oiltype' ])
te = enc.fit_transform(X_encoder, y_encoder)


# In[6]:


te.head()


# In[7]:


train_data.head()


# In[8]:


from sklearn.preprocessing import MinMaxScaler
#特征归一化

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(pd.concat([te]).values)
all_data = min_max_scaler.transform(pd.concat([te]).values)


# In[9]:


from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA
pca = RandomizedPCA(20)
pca.fit(all_data)
X = all_data
y = train_data['price'] .values


# In[10]:


all_data


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[12]:


import numpy as np 
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[13]:


# # 6.MLP导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname = "/Library/Fonts/华文细黑.ttf",size=14)

from sklearn import metrics
from sklearn.model_selection import train_test_split
## 忽略提醒
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
#PCA
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA
#多元线性回归分析
train_xadd = sm.add_constant(X_train)  ## 添加常数项
lm = sm.OLS(y_train,train_xadd).fit()
lm.summary()


# In[81]:


#检查模型再测试集上的预测效果
test_xadd = sm.add_constant(X_test)  ## 添加常数项
pre_y_xadd = lm.predict(test_xadd)
print("mean absolute error:", metrics.mean_absolute_error(y_test,pre_y_xadd))
print("mean squared error:", metrics.mean_squared_error(y_test,pre_y_xadd))


# In[82]:


# 结果输出
with open('result1.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the data
    writer.writerow(pre_y_xadd)


# In[51]:


## 定义含有4个隐藏层的MLP网络
mlpr = MLPRegressor(hidden_layer_sizes=(1000,1000,300,100), ## 隐藏层的神经元个数
                    activation='tanh', 
                    solver='adam', 
                    alpha=0.0001,   ## L2惩罚参数
                    max_iter=100, 
                    random_state=123,
#                     early_stopping=True, ## 是否提前停止训练
#                     validation_fraction=0.2, ## 20%作为验证集
#                     tol=1e-8,
                   )

## 拟合训练数据集
mlpr.fit(X_train,y_train)

## 可视化损失函数
plt.figure()
plt.plot(mlpr.loss_curve_)
plt.xlabel("iters")
plt.ylabel(mlpr.loss)
plt.show()


# In[52]:


# 相对误差
def mean_relative_error(y_true, pre_y_mlp,):
    import numpy as np
    relative_error = np.average(np.abs(y_true - y_pred) / y_true, axis=0)
    return relative_error


# In[54]:


## 对测试集上进行预测
pre_y_mlp = mlpr.predict(X_test)
print("mean absolute error:", metrics.mean_absolute_error(y_test,pre_y_mlp))
print("mean squared error:", metrics.mean_squared_error(y_test,pre_y_mlp))

## 输出在测试集上的R^2
print("在训练集上的R^2:",mlpr.score(X_train,y_train))
print("在测试集上的R^2:",mlpr.score(X_test,y_test))


# In[56]:


import csv
pre_y_all = mlpr.predict(X)
# 结果输出
with open('result1.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the data
    writer.writerow(pre_y_all)


# In[84]:


#基于keras的MLP分析
from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(units=1000, input_dim=X_train.shape[1], 
                kernel_initializer='uniform', 
                activation='relu'))

model.add(Dense(units=1000, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=300, 
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=300, 
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=100, 
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='linear'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.summary()


# In[45]:


# k折验证
k = 4
num_val_samples = len(X_train) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]  # 准备验证数据：第k个分区的数据
    val_targets = y_train[i * num_val_samples:(i + 1) * num_val_samples]
    # 准备训练数据：其他所有分区的数据
    partial_train_data = np.concatenate(
        [X_train[:i * num_val_samples],
         X_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = model
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,
              verbose=0)  # 训练模型（静默模式，vervose=0）
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)  # 在验证数据上评估模型
    all_scores.append(val_mae)
print(np.mean(all_scores))
# 保存每折的验证结果
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]  # 准备验证数据：第k个分区的数据
    val_targets = y_train[i * num_val_samples:(i + 1) * num_val_samples]
    # 准备训练数据：其他所有分区的数据
    partial_train_data = np.concatenate(
        [X_train[:i * num_val_samples],
         X_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = model
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# 绘制验证分数（验证）

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# In[79]:


from keras.callbacks import EarlyStopping
## 提前停止条件
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.compile(loss='mean_squared_error', optimizer='adam')
model_fit = model.fit(X_train, y_train,batch_size=16,
                      epochs=200, verbose=0,
                      validation_split=0.2,
                      callbacks=[early_stopping])


# In[80]:



## 对测试集上进行预测
pre_y = model.predict(X_test)
print("mean absolute error:", metrics.mean_absolute_error(y_test,pre_y))
print("mean squared error:", metrics.mean_squared_error(y_test,pre_y))


# In[82]:


from sklearn.metrics import r2_score
print('accuracy=', r2_score(pre_y,y_test))


# In[67]:


import numpy as np
np.savetxt('result3.csv',(pre_y_all) ,delimiter=',')


# In[55]:


all_data_list = [all_data]
import csv
pre_y_all = model.predict(all_data)
# 结果输出
import numpy as np
np.savetxt('result1.csv',(pre_y_all) ,delimiter=',')


# In[39]:


import csv
test_data = pd.read_csv('old3.csv')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders.target_encoder import TargetEncoder
import numpy as np
import pandas as pd
XX= test_data.drop(['carid', 'price'], axis=1)
yy = test_data['carid']
encc = TargetEncoder(cols=[ 'color', 'cityId', 'transferCount',  'country', 'displacement', 'gearbox',
       'oiltype' ])
tee = enc.fit_transform(XX, yy)


# In[40]:


tee.head()


# In[41]:


tee.info()


# In[42]:


from sklearn.preprocessing import MinMaxScaler
#特征归一化
XXX = te
yyy = test_data['price'] 
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(pd.concat([tee]).values)                       
all_dataa = min_max_scaler.transform(pd.concat([tee]).values)           


# In[44]:


#pca
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA
pca = RandomizedPCA(20)
pca.fit(all_dataa)
#绘制曲线
import numpy as np 
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
pre_y = model.predict(all_dataa)


# In[45]:


len(pre_y)


# In[56]:


# 结果输出
import numpy as np
np.savetxt('result1.csv',(pre_y) ,delimiter=',')


# In[ ]:




