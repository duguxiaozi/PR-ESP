import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from Dataprocessing import Xmoodlabel, X

clf_1 = []
rfc_1 = []

for i in range(8):
    rfc = RandomForestClassifier(n_estimators=87)
    rfc_s = cross_val_score(rfc, X, Xmoodlabel.ravel(), cv=i+2).mean()
    rfc_1.append(rfc_s)

    clf = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=6)
    clf_s = cross_val_score(clf, X, Xmoodlabel.ravel(), cv=i+2).mean()
    clf_1.append(clf_s)

plt.plot(range(3, 11), rfc_1, label="Random Forest")
plt.plot(range(3, 11), clf_1, label="Decision Forest")
plt.xlabel('cv')
plt.legend()
plt.show()