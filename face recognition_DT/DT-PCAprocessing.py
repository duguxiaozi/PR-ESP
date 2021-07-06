from sklearn import tree
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Dataprocessing import X, Xmoodlabel

#PCA降维后的决策树准确率图像
Pca = []
for i in range(100):
    pca = PCA(n_components=i+1)
    X_pca = pca.fit_transform(X)
    xtrain_pca,xtest_pca,ytrain,ytest=train_test_split(X_pca,Xmoodlabel,test_size=0.3)
    #train
    clf = tree.DecisionTreeClassifier(criterion="entropy",splitter="random",max_depth=6)
    clf.fit(xtrain_pca, ytrain.ravel())
    #test
    clf_s =clf.score(xtest_pca,ytest.ravel())
    Pca.append(clf_s)
print(max(Pca),Pca.index(max(Pca)))
plt.figure(figsize=[20,5])
plt.plot(range(1,101),Pca)
plt.title('PCA of Decision trees')
plt.show()

