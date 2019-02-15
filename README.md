### 机器学习
#### 分类
主要分为几个大类：
* 监督学习(数据中带有我们想要预测的附加属性)
  * 分类：样本属于两个或更多类，我们从标记的数据中学习预测未标记数据的类别，如手写数字识别
  * 回归: 期望的输出由一个或连续变量组成(函数)，如预测鱼的长度和其年龄和体重的函数
* 无监督学习(训练数据中只有输入向量，没有任何目标值)
  * 聚类：在数据中发现类似的组
  * 密度估计：确定输入空间内的数据分布
  * 高维数据投影数据空间缩小到两维或三维进行可视化
  
#### 数据
训练集，测试集，28原则

#### scikit-learn
```python
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris() # 虹膜
digits = datasets.load_digits() # 数字

print(digits.data) # 数据集
print(digits.target) # 数据集的真实数据
print(digits.images[0]) # 原始数据的图像形状

clf = svm.SVC(gamma=0.001, C=100.) # 参数, 使用网格搜索和交叉验证等工具，自动找到参数的良好值
clf.fit(digits.data[:-1], digits.target[:-1]) # 训练
clf.predict(digits.data[-1:])

from sklearn.externals import joblib
joblib.dump(clf, 'svc.pkl') # 持久化

clf = joblib.load('svc.pkl')

clf.set_params(kernel='linear').fix() # 更新参数
clf.predict()

```
