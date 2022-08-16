from RCS_ML_prepare import read_data
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np


signal_file = r'RCSsequence'
label_file = r'RCS_label'
num_classes = 8  # 类别数
ntrain = 300  # 选择每类训练样本数量
sampling_point = 2000   # RCS为2000，一维距离像为512

# 数据准备
(x_train, y_train), (x_test, y_test) = read_data(signal_file, label_file, ntrain_per_class=ntrain)

# 模型搭建及训练
clf = SVC(C=0.001, gamma=0.1)   # 默认为rbf核函数，可调参数c,gamma
clf.fit(x_train.reshape([-1, sampling_point]), np.argmax(y_train, 1))

# 测试
train_acc = clf.score(x_train.reshape([-1, sampling_point]), np.argmax(y_train, 1))
test_acc = clf.score(x_test.reshape([-1, sampling_point]), np.argmax(y_test, 1))
pred = clf.predict(x_test.reshape([-1, sampling_point]))
pred_acc = np.mean(pred == np.argmax(y_test, 1))

# 输出训练和测试集准确率，测试集混淆矩阵
print("Accuracy on training set:", train_acc)
print("Accuracy on test set:", test_acc)
print("predict accuracy:", pred_acc)
print('confusion_matrix:\n', confusion_matrix(np.argmax(y_test, 1), pred))
