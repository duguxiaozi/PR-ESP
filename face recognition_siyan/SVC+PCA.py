
#########这个程序是SVM的分类器，模型是SVC(kernel='linear', C=1, gamma=1)，特征提取方法是PCA提取
import numpy as np
import matplotlib.pyplot as plt
from time import time
xdata =np.load('../Xtrain.npy')
xlabel=np.load('../xlabel.npy')
xmood = xlabel[:,[8]]
xmoodlabel = np.ones((3991,1))
for i in range(3991):
    a= int(xmood[i,:])
    xmoodlabel[i,:]=a
####################pca降维
n_samples,h,w = xdata.shape
X = xdata.reshape(n_samples,h*w)
y = xmoodlabel.ravel()
from sklearn.decomposition import PCA
n_class = 3
n_components = 120
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X.shape[0]))
time1 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
x = pca.transform(X)
##########################分类算法
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
###########################随机划分
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
###########################K折交叉验证
from sklearn.model_selection import KFold
skf = KFold(n_splits = 5)
scores1 = []
scores2 =[]
a= 0
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    a= a+1
    ##############################################将最佳的参数直接代入减少运行时间
    svc = SVC(kernel='linear', C=1, gamma=1)
    svc_fit = OneVsRestClassifier((svc)).fit(x_train, y_train)
    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(svc_fit, x_train, y_train)
    plt.show()
    plot_confusion_matrix(svc_fit, x_test, y_test)
    plt.show()
    scores4 = svc_fit.score(x_test, y_test)
    scores2.append(scores4)
    scores3 = svc_fit.score(x_train, y_train)
    scores1.append(scores3)
    print("The %s fold Accuracy of the SVC algorithm training set: %0.2f (+/- %0.2f)" % (a,scores3.mean(), scores3.std() * 2))
    print("The %s fold Accuracy of the SVC algorithm test set: %0.2f (+/- %0.2f)" % (a,scores4.mean(), scores4.std() * 2))
print("pca+svm mean training data Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores1), np.std(scores1) * 2))
print("pca+svm mean testing data Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores2), np.std(scores2) * 2))
print('pca+svm running time %s'%(time()-time1))