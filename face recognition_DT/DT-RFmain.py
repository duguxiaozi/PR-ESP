import matplotlib.pyplot as plt
from Dataprocessing import X_Std_train, Ytrain, Xmoodlabel, X
from sklearn import tree

# 决策树拟合情况
# 训练
clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=6)
clf.fit(X_Std_train, Ytrain.ravel())
# 测试
score = clf.score(X_Std_train, Ytrain.ravel())
print('预测准确率：', score)

# 随机森林拟合情况
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier()
rfc.fit(X_Std_train, Ytrain.ravel())
# 测试
score1 = rfc.score(X_Std_train, Ytrain.ravel())
print('预测准确率：', score1)

#随机森林参数n_estimators的整定
superpa = []
for i in range(100):
    rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
    rfc_s = cross_val_score(rfc,X,Xmoodlabel.ravel(),cv=3).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa)))
plt.figure(figsize=[20,5])
plt.plot(range(1,101),superpa)
plt.show()

# 单一决策树与随机森林比较
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

xtrain, xtest, ytrain, ytest = train_test_split(X, Xmoodlabel, test_size=0.3)

clf = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=6)
rfc = RandomForestClassifier(n_estimators = 87)
clf = clf.fit(xtrain, ytrain.ravel())
rfc = rfc.fit(xtrain, ytrain.ravel())
score_c = clf.score(xtest, ytest.ravel())
score_r = rfc.score(xtest, ytest.ravel())
print("Decision Tree:{}".format(score_c)
      , "Random Forest:{}".format(score_r)
      )
