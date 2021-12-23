# 整个思路 模型脱敏（结构化）通过categorical——embedder方式——神经网络，自行搭建测试——k折验证，自主调参——(可以加正则化优化)
# 导入数据
import pandas as pd
import numpy as np
import categorical_embedder as ce
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
import matplotlib.pyplot as plt

trian_data = pd.read_csv('train_estimate.csv')

# 数据脱敏(one-hot-encoding)
X = trian_data.drop(['Column1','Column36'],axis=1)
y = trian_data['Column36']

# 确定分类变量
embedding_info = ce.get_embedding_info(X)
print(embedding_info)
# 整数编码
X_encoded,encoders = ce.get_label_encoded_data(X)
print(X_encoded,encoders)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y)

# ce.get_embeddings trains NN, extracts embeddings and return a dictionary containing the embeddings
embeddings = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info,is_classification=True, epochs=100,batch_size=256)
print(embeddings)
trian_data_regular = ce.fit_transform(X,embeddings=embeddings, encoders=encoders,drop_categorical_vars=True)
X_regular = trian_data_regular.drop(['Column1','Column36'],axis=1)
y_regualr = trian_data_regular['Column36']
# 构建网络

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(X.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1)) #线性
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
# k折验证
k=4
num_val_samples = len(X_regular) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #',i)
    val_data = X_regular[i * num_val_samples: (i + 1) * num_val_samples] #准备验证数据：第k个分区的数据
    val_targets = y_regualr[i * num_val_samples:(i + 1) * num_val_samples]
    # 准备训练数据：其他所有分区的数据
    partial_train_data = np.concatenate(
        [X_regular[:i * num_val_samples],
         X_regular[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_regualr[:i * num_val_samples],
         y_regualr[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=0) #训练模型（静默模式，vervose=0）
    val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)#在验证数据上评估模型
    all_scores.append(val_mae)
print(np.mean(all_scores))
# 保存每折的验证结果
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #',i)
    val_data = X_regular[i * num_val_samples: (i + 1) * num_val_samples] #准备验证数据：第k个分区的数据
    val_targets = y_regualr[i * num_val_samples:(i + 1) * num_val_samples]
    # 准备训练数据：其他所有分区的数据
    partial_train_data = np.concatenate(
        [X_regular[:i * num_val_samples],
         X_regular[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_regualr[:i * num_val_samples],
         y_regualr[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=num_epochs,batch_size=1,verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append((mae_history))
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# 绘制验证分数（验证）

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
#绘制验证分数
def smooth_curve(points, factor=0.9):
 smoothed_points = []
 for point in points:
    if smoothed_points:
        previous = smoothed_points[-1]
        smoothed_points.append(previous * factor + point * (1 - factor))
    else:
        smoothed_points.append(point)
 return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#训练最终模型
model = build_model()
model.fit(X_regular,y_regualr,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score = model.evaluate(X_regular,y_regualr)
print(test_mse_score)