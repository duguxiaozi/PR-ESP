
#########这个程序是SVM的分类器，模型是SVC(kernel='linear', C=1, gamma=1)，特征提取方法是LBP提取
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import feature as skft
xdata =np.load('../Xtrain.npy')
xlabel=np.load('../xlabel.npy')
xmood = xlabel[:,[8]]
xmoodlabel = np.ones((3991,1))
for i in range(3991):
    a= int(xmood[i,:])
    xmoodlabel[i,:]=a
##########################LBP特征提取
X =xdata
y = xmoodlabel.ravel()
radius =1
n_point = 8 * radius
def texture_detect(X):
    x = np.zeros((3991,256))
    for i in np.arange(3991):
        #使用LBP方法提取图像的纹理特征.
        lbp=skft.local_binary_pattern(X[i],n_point,radius,'default')
        #统计图像的直方图
        # lbp2 = lbp.astype(np.int32)
        max_bins = int(lbp.max() + 1)
        # #hist size:256
        x[i], _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins))
    return x
time1 = time()
x = texture_detect(X)
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