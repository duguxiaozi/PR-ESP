#选择最好的n_components:累积可解析方差贡献率曲线
#pca_line = PCA().fit(X)
#plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
#plt.xticks()
#plt.xlabel("number of components after dimension reduction")
#plt.ylabel("cumulative explained variance ratio")
#plt.show()

#PCA降维后的随机森林准确率图像
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from Dataprocessing import Xmoodlabel, X

Pca1 = []
for j in range(100):
    pca = PCA(n_components=j+1)
    X_pca = pca.fit_transform(X)
    xtrain_pca,xtest_pca,ytrain,ytest=train_test_split(X_pca,Xmoodlabel,test_size=0.3)
    #train
    rfc = RandomForestClassifier(n_estimators = 87)
    rfc.fit(xtrain_pca, ytrain.ravel())
    #test
    rfc_s =rfc.score(xtest_pca,ytest.ravel())
    Pca1.append(rfc_s)
print(max(Pca1),Pca1.index(max(Pca1)))
plt.figure(figsize=[20,5])
plt.plot(range(1,101),Pca1)
plt.title('PCA of Random forest')
plt.show()