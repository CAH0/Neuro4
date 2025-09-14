import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd



###############################################################################
###############################################################################
# Теперь посмотрим что изменится в структуре нейронной сети, если нам нужно решить задачу регрессии
# Задача регрессии заключается в предсказании значений одной переменной
# по значениям другой (других).  
# От задачи классификации отличается тем, что выходные значения нейронной сети не 
# ограничиваются значениями меток классов (0 или 1), а могут лежать в любом 
# диапазоне чисел.
# Примерами такой задачи можгут быть предсказание цен на жилье, курсов валют или акций,
# количества выпадающих осадков или потребления электроэнергии.

# Рассмотрим задачу предсказания прочности бетона (измеряется в мегапаскалях)
df = pd.read_csv('dataset_simple.csv')

# Известно что прочность бетона зависит от многих факторов - количесва цемента, 
# используемых добавок, 
# Cement - количество цемента в растворе kg/m3
# Blast Furnace Slag - количество шлака в растворе kg/m3 
# Fly Ash - количетво золы в растворе kg/m3
# Water - количетво воды в растворе kg/m3
# Superplasticizer - количетво пластификатора в растворе kg/m3
# Coarse Aggregate - количетво крупного наполнителя в растворе kg/m3
# Fine Aggregate - количетво мелкого наполнителя в растворе kg/m3
# Age - возраст бетона в днях
# Concrete compressive strength -  прочность бетона MPa


X = torch.Tensor(df[['age']].values)  # 2. Берем только столбец с возрастом как признак
y = torch.Tensor(df['income'].values).reshape(-1, 1)  # 3. Берем доход как целевую переменную

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, -1].values, marker='o')

# Чтобы выходные значения сети лежали в произвольном диапазоне,
# выходной нейрон не должен иметь функции активации или 
# фуннкция активации должна иметь область значений от -бесконечность до +бесконечность

class NNet_regression(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size) # просто сумматор
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = X.shape[1] # количество признаков задачи 
hiddenSizes = 3   #  число нейронов скрытого слоя 
outputSize = 1 # число нейронов выходного слоя

net = NNet_regression(inputSize,hiddenSizes,outputSize)

# В задачах регрессии чаще используется способ вычисления ошибки как разница квадратов
# как усредненная разница квадратов правильного и предсказанного значений (MSE)
# или усредненный модуль разницы значений (MAE)
lossFn = nn.L1Loss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

epohs = 1
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred.squeeze(), y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%1==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

print('\nПредсказания:') # Иногда переобучается, нужно запускать обучение несколько раз
print(pred[0:10])
err = torch.mean(abs(y - pred.T).squeeze()) # MAE - среднее отклонение от правильного ответа
print('\nОшибка (MAE): ')
print(err) # измеряется в MPa


# Построим график
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, -1].values, marker='o')

with torch.no_grad():
    y1 = net.forward(torch.Tensor([100]))
    y2 = net.forward(torch.Tensor([600]))

plt.plot([100,600], [y1.numpy(),y2.numpy()],'r')

