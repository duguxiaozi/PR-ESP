import numpy as np
Xdata = np.load('../Xtrain.npy')
Xlabel = np.load('../Xlabel.npy')

Xmood = Xlabel[:,[8]]
Xmoodlabel=np.ones((3991,1))
for i in range(3991):
    a=int(Xmood [i,:])
    Xmoodlabel[i,:]=a

X=np.ones((3991,16384))
for i in range(3991):
    a=Xdata[i].flatten()
    X[i,:]=a
Xtrain=X[0:3550,:]
Ytrain=Xmoodlabel[0:3550,:]
Xtest=X[3551:3990,:]
Ytest=Xmoodlabel[3551:3990,:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Std_train = sc.fit_transform(Xtrain)
X_Std_test = sc.fit_transform(Xtest)