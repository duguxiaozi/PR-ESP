
#################这个搜索跑的太慢了，需要跑很长的时间，如果第一个搜索模块不行的话，运行第二个搜索模块
import numpy as np
from time import time
xdata =np.load('../Xtrain.npy')
xlabel=np.load('../xlabel.npy')
xmood = xlabel[:,[8]]
xmoodlabel = np.ones((3991,1))
for i in range(3991):
    a= int(xmood[i,:])
    xmoodlabel[i,:]=a
n_samples,h,w = xdata.shape
X = xdata.reshape(n_samples,h*w)
y = xmoodlabel.ravel()
from skimage import feature as skft
####################pca降维
from sklearn.decomposition import PCA
n_class = 3
n_components = 120
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
x = pca.transform(X)
##########################分类算法
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
skf = KFold(n_splits = 5)
scores1 = []
scores2 =[]
a=0
#########################################网络搜索模块
for train_index, test_index in skf.split(x, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    from sklearn.model_selection import GridSearchCV
    parameters = {'estimator__kernel': ('linear', 'rbf', 'poly'), 'estimator__C': [1, 10, 100, 1000], 'estimator__gamma': [1, 0.1, 0.01, 0.01]}
    svc_model = OneVsRestClassifier(SVC())
    svc_grid = GridSearchCV(svc_model, param_grid=parameters,scoring = 'accuracy')
    SVC_fit = svc_grid.fit(x_train, y_train)
    print('Best model estimator：{}'.format(svc_grid.best_estimator_))
    scores4 = svc_grid.best_estimator_.score(x_test, y_test)
    scores2.append(scores4)
    scores3 = svc_grid.best_estimator_.score(x_train, y_train)
    scores1.append(scores3)
    a = a+1
    print("The %i fold best_KNN training Accuracy: %0.2f " % (a,scores3.mean()))
    print("The %i fold best_KNN testing Accuracy: %0.2f " % (a,scores4.mean()))
    print('the best model accuracy of training set %0.2f'%(svc_grid.score(x_train,y_train)))
    print('the best model accuracy of testing set %0.2f' % (svc_grid.score(x_test, y_test)))
print("Training Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores2), np.std(scores2) * 2))
print("Testing Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores2), np.std(scores2) * 2))
###########################################################另外一个搜索模块
# for train_index, test_index in skf.split(x, y):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.model_selection import GridSearchCV
#     parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.01]}
#     svc_model = SVC(decision_function_shape= 'ovr')
#     svc_rbf = GridSearchCV(svc_model, parameters,cv = 2).fit(x_train, y_train)
#     print('最佳模型的估计器设置：{}'.format(svc_rbf.best_estimator_))
#     # print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
#     print('最佳模型的准确率设置：{}'.format(svc_rbf.best_estimator_.score(x_test,y_test)))
# print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores2), np.std(scores2) * 2))
