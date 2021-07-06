import numpy as np
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from time import time
from sklearn.model_selection import KFold

test_number = 399 #总样本为3990，,10次交叉验证，每次交叉验证的样本为399

t0 = time()

#读取卷标信息
filename = "../faceDR"
target = np.zeros([3991, 8])
row = 0
with open(filename, "r") as ins:
    for line in ins:
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line[6] == "m":
            continue
        elif line[0] == "1" or line[0] == "2" or line[0] == "3":

            target[row][0] = line[0]
            target[row][1] = line[1]
            target[row][2] = line[2]
            target[row][3] = line[3]

            sex_position = line.index("_s")
            if line[sex_position+4] == "m":
                target[row][4] = 1
            else:
                target[row][4] = -1

            age_position = line.index("_a")
            if line[age_position+4] == "c":
                target[row][5] = 1
            elif line[age_position+4] == "t":
                target[row][5] = 2
            elif line[age_position+4] == "a":
                target[row][5] = 3
            else:
                target[row][5] = 4

            race_position = line.index("_r")
            if line[race_position+5] == "w":
                target[row][6] = 1
            elif line[race_position+5] == "a":
                target[row][6] = 2
            else:
                target[row][6] = 3

            face_position = line.index("_f")
            if line[face_position+6] == "m":
                target[row][7] = 1
            elif line[face_position+6] == "e":
                target[row][7] = 2
            else:
                target[row][7] = 3

            row = row + 1
        else:
            continue

#读取卷标信息
filename = "../faceDS"
with open(filename, "r") as ins:
    for line in ins:
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line[6] == "m":
            continue
        elif line[0] == "3" or line[0] == "4" or line[0] == "5":

            target[row][0] = line[0]
            target[row][1] = line[1]
            target[row][2] = line[2]
            target[row][3] = line[3]

            sex_position = line.index("_s")
            if line[sex_position+4] == "m":
                target[row][4] = 1
            else:
                target[row][4] = -1

            age_position = line.index("_a")
            if line[age_position+4] == "c":
                target[row][5] = 1
            elif line[age_position+4] == "t":
                target[row][5] = 2
            elif line[age_position+4] == "a":
                target[row][5] = 3
            else:
                target[row][5] = 4

            race_position = line.index("_r")
            if line[race_position+5] == "w":
                target[row][6] = 1
            elif line[race_position+5] == "a":
                target[row][6] = 2
            else:
                target[row][6] = 3

            face_position = line.index("_f")
            if line[face_position+6] == "m":
                target[row][7] = 1
            elif line[face_position+6] == "e":
                target[row][7] = 2
            else:
                target[row][7] = 3

            row = row + 1
        else:
            continue


#读取mat格式的图像文件(原图像文件已在matlab中处理成mat格式)
data = np.zeros([3991, 16384])
row = 0
for filename in os.listdir("../rawdatamat"):
    newname = '../rawdatamat/' + filename
    a = sio.loadmat(newname)
    a = a['I']
    for i in range(0, 16384):
        data[row][i] = a[i]

    row = row+1


#选择表情作为此次分类的对象
y = np.zeros([3991, 1])
for i in range(0, 3991):
    y[i] = target[i][7]
y = y.ravel()

#为了方便做10次交叉验证，3991个样本减少1个，变为3990个
data = data[:-1]
y = y[:-1]

#统计每类表情个数
a = b = c = 0
for i in range(0, 3990):
    if y[i] == 1:
        a += 1
    elif y[i] == 2:
        b += 1
    elif y[i] == 3:
        c += 1
print("第一类表情个数：", a)
print("第二类表情个数：", b)
print("第三类表情个数：", c)
#x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=test_number)

kf = KFold(n_splits=10, shuffle=True, random_state=0) #10次交叉验证

j = 1
sum_error1 = 0
sum_error2 = 0
sum_error3 = 0
sum_error = 0
sum_correct1 = 0
sum_correct2 = 0
sum_correct3 = 0
sum_correct = 0

for train_index, test_index in kf.split(data, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("这是第%d次交叉验证" % j)
    j = j+1

    n_components = 150
    pca = PCA(n_components=n_components, whiten=True).fit(x_train) #PCA降维处理

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    clf.decision_function_shape = "ovo" #选择ovo或ovr模式
    clf = clf.fit(x_train_pca, y_train) #模型训练

    y_pred = clf.predict(x_test_pca)

    #print(y_test)
    #print(y_pred)

    print("done in %0.3fs" % (time() - t0)) #运行时间

    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct = 0
    error1 = 0
    error2 = 0
    error3 = 0
    error = 0

    #统计每类表情预测的正确个数和错误的个数
    for i in range(0, test_number):
        if y_test[i] == 1 and y_pred[i] == 1:
            correct1 += 1
        elif y_test[i] == 1 and y_pred[i] != 1:
            error1 += 1
        elif y_test[i] == 2 and y_pred[i] == 2:
            correct2 += 1
        elif y_test[i] == 2 and y_pred[i] != 2:
            error2 += 1
        elif y_test[i] == 3 and y_pred[i] == 3:
            correct3 += 1
        elif y_test[i] == 3 and y_pred[i] != 3:
            error3 += 1

    correct = correct1+correct2+correct3
    error = error1+error2+error3

    #统计总的正确个数和错误个数
    sum_correct1 += correct1
    sum_correct2 += correct2
    sum_correct3 += correct3
    sum_correct += correct
    sum_error1 += error1
    sum_error2 += error2
    sum_error3 += error3
    sum_error += error

    #打印相关结果
    print("第一类正确率为：", correct1 / (correct1 + error1))
    print("第一类错误率为：", error1 / (correct1 + error1))
    print("第二类正确率为：", correct2 / (correct2 + error2))
    print("第二类错误率为：", error2 / (correct2 + error2))
    print("第三类正确率为：", correct3 / (correct3 + error3))
    print("第三类错误率为：", error3 / (correct3 + error3))
    print("全部正确率为：", correct / (correct + error))
    print("全部错误率为：", error / (correct + error))

print("第一类总正确率为：", sum_correct1/(sum_correct1+sum_error1))
print("第一类总错误率为：", sum_error1/(sum_correct1+sum_error1))
print("第二类总正确率为：", sum_correct2/(sum_correct2+sum_error2))
print("第二类总错误率为：", sum_error2/(sum_correct2+sum_error2))
print("第三类总正确率为：", sum_correct3/(sum_correct3+sum_error3))
print("第三类总错误率为：", sum_error3/(sum_correct3+sum_error3))
print("全部总正确率为：", sum_correct/(sum_correct+sum_error))
print("全部总错误率为：", sum_error/(sum_correct+sum_error))











