import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from time import time
xdata =np.load('../Xtrain.npy')
xlabel=np.load('../xlabel.npy')
xmood = xlabel[:,[8]]
xmoodlabel = np.ones((3991,1))
for i in range(3991):
    a= int(xmood[i,:])
    xmoodlabel[i,:]=a
####################pca降维
from sklearn.decomposition import PCA
start_time = time()
n_class = 3               ###############目标分类是3类，这个赋值没有作用
############################数据标准
from sklearn import preprocessing
n_samples,h,w = xdata.shape
X = xdata.reshape(n_samples,h*w)
time1 = time()
scaler1 = preprocessing.StandardScaler().fit(X)
X = scaler1.transform(X)
print('standardSclaer time :%s '%(time()-time1))
y = xmoodlabel.ravel()
n_components = 120
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X.shape[0]))
pca = PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(X)
x = pca.transform(X)
end_time = time()
print ('PCA time：%s'%(end_time-start_time))
##########################分类算法
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
###########################K折交叉验证
skf = KFold(n_splits = 5)
scores1 = []
scores2 =[]
scores5 = []
scores6 =[]
scores9 = []
scores10 =[]
scores13 = []
scores14 =[]
a = 0
##############################################寻找最佳的参数
for train_index, test_index in skf.split(x, y):
    #################可以看出K折划分的具体标签
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    parameters = {'estimator__n_neighbors':[i for i in range(1,20)]}
    knn_model = OneVsRestClassifier(KNeighborsClassifier(algorithm='auto'))
    knn_grid = GridSearchCV(knn_model,param_grid=parameters,scoring = 'accuracy')
    knn_griD = knn_grid.fit(x_train, y_train)
    #################################注释解开可以看出寻找参数过程的每一次结果
    # means = knn_grid.cv_results_['mean_test_score']
    # stds = knn_grid.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, knn_grid.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #             % (mean, std * 2, params))
    # print()
    print('最佳模型的参数设置：{}'.format(knn_grid.best_params_))
    scores12 = knn_grid.best_estimator_.score(x_test, y_test)
    scores10.append(scores12)
    scores11 = knn_grid.best_estimator_.score(x_train, y_train)
    scores9.append(scores11)
    print('最佳模型的测试集准确率：{}'.format(knn_grid.best_estimator_.score(x_test,y_test)))
    KNN = OneVsRestClassifier(KNeighborsClassifier( n_neighbors = 10, algorithm = 'auto'))
    knn = KNN.fit(x_train, y_train)
    bagging = BaggingClassifier(KNN, max_samples=0.6, max_features=0.3)
    #####################这里是混淆矩阵部分，注释解开可以画每一次K折的混淆矩阵，模型是：k=10的KNN
    # from sklearn.metrics import plot_confusion_matrix
    # plot_confusion_matrix(knn, x_train, y_train)
    # plt.show()
    # plot_confusion_matrix(knn, x_test, y_test)
    # plt.show()
    #####################这里是混淆矩阵部分，注释解开可以画每一次K折的混淆矩阵，模型是：k=10的bagging
    # from sklearn.metrics import plot_confusion_matrix
    # plot_confusion_matrix(bagging.fit(x_train,y_train), x_train, y_train)
    # plt.show()
    # plot_confusion_matrix(bagging.fit(x_train,y_train), x_test, y_test)
    # plt.show()
    ##################################混淆矩阵，模型是：k=1-20的最好的KNN
    # from sklearn.metrics import plot_confusion_matrix
    # plot_confusion_matrix(knn_grid, x_train, y_train)
    # plt.show()
    # plot_confusion_matrix(knn_grid, x_test, y_test)
    # plt.show()
    scores7 = OneVsRestClassifier(bagging).fit(x_train, y_train).score(x_train, y_train)
    scores5.append(scores7)
    scores8 = OneVsRestClassifier(bagging).fit(x_train, y_train).score(x_test, y_test)
    scores6.append(scores8)
    scores4 = knn.score(x_test, y_test)
    scores2.append(scores4)
    scores3 = OneVsRestClassifier((knn)).fit(x_train, y_train).score(x_train, y_train)
    scores1.append(scores3)
    a = a+1
    print("The %i fold best_KNN training Accuracy: %0.2f (+/- %0.2f)" % (a,scores11.mean(), scores11.std() * 2))
    print("The %i fold best_KNN testing Accuracy: %0.2f (+/- %0.2f)" % (a,scores12.mean(), scores12.std() * 2))
    print("The %i fold KNN=10 training Accuracy: %0.2f (+/- %0.2f)" % (a,scores3.mean(), scores3.std() * 2))
    print("The %i fold KNN=10 testing Accuracy: %0.2f (+/- %0.2f)" % (a,scores4.mean(), scores4.std() * 2))
    print("The %i fold bagging training Accuracy: %0.2f (+/- %0.2f)" % (a,scores7.mean(), scores7.std() * 2))
    print("The %i fold bagging testing Accuracy: %0.2f (+/- %0.2f)" % (a,scores8.mean(), scores8.std() * 2))
print("KNN=10 mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores2), np.std(scores2) * 2))
print("bagging mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores6), np.std(scores6) * 2))
print("Best_knn mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores10), np.std(scores10) * 2))