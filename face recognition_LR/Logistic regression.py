import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
Xdata = np.load('../Xtrain.npy')
Xlabel = np.load('../Xlabel.npy')
Xmood = Xlabel[:, 8]
X_data = np.reshape(Xdata, (-1, 16384)) #将三维数组处理成二维数组，每一行储存一张图片
standard_data = StandardScaler().fit_transform(X_data)#将数据进行标准化处理，处理成standard_data
mood = list(map(int, Xmood))

####生成原始数据original_data
X_original_data = np.column_stack((standard_data, mood))
original_data = pd.DataFrame(X_original_data)  #将数据处理成DataFrame格式，便于后续利用pandas库来对数据进行处理
Names = []
for i in range(1, 16385):
    Names.append('V' + str(i))
Names.append('Class')
original_data.columns = Names
original_data['Class'] = original_data['Class'].astype(np.int)

####生成降维数据PCA_data,选择的维度是641维
from sklearn.decomposition import PCA
pca = PCA(n_components=641, svd_solver='auto', whiten=True).fit(standard_data) #0.95对应的维度是162维，0.98对应的维度是390维
XP_data = pca.transform(standard_data)
X_PCA_data = np.column_stack((XP_data, mood))#将降维后的图像数据和标签合成一个数据集合
PCA_data = pd.DataFrame(X_PCA_data)  #将数据处理成DataFrame格式，便于后续利用pandas库来对数据进行处理
Name = []
for i in range(1, 642):
    Name.append('V' + str(i))
Name.append('Class')
PCA_data.columns = Name
PCA_data['Class'] = PCA_data['Class'].astype(np.int)

#以上的两个操作为后续模型的建立提供了两种数据 <original_data> <PCA_data>

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings("ignore")

#定义通过交叉验证最佳正则化系数C函数
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(10, shuffle=False)
    # 定义不同力度的正则化惩罚力度
    c_param_range = [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
    # 展示结果用的表格
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # k-fold 表示K折的交叉验证，这里会得到两个索引集合: 训练集 = indices[0], 验证集 = indices[1]
    j = 0
    # 循环遍历不同的参数
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('Regularization coefficient C: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        # 一步步分解来执行交叉验证
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # 指定算法模型，并且给定参数
            lr0 = LogisticRegression(C=c_param, penalty='l1', solver='liblinear', class_weight='balanced')
            lr = OneVsRestClassifier(lr0)   #OVR模型
            #lr = OneVsOneClassifier(lr0)   #OVO模型

            # 训练模型，注意索引不要给错了，训练的时候一定传入的是训练集，所以X和Y的索引都是0
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            # 建立好模型后，预测模型结果，这里用的就是验证集，索引为1
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # 有了预测结果之后就可以来进行评估了，这里计算正确率需要传入预测值和真实值。<因为是多分类问题故需要修改average的值average='micro'>
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample, average='micro')
            # 一会还要算平均，所以把每一步的结果都先保存起来。
            recall_accs.append(recall_acc)
            print('Iteration ', iteration+1, ': accuracy = ', recall_acc)

        # 当执行完所有的交叉验证后，计算平均准确率结果
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Average accuracy: ', np.mean(recall_accs))
        print('')

    # 找到最好的参数，哪一个正确率高，自然就是最好的了。
    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']

    # 打印最好的结果
    print('*********************************************************************************')
    print('Best regularization coefficient C = ', best_c)
    print('*********************************************************************************')

    return best_c

#绘制混淆矩阵函数
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

######################################################
###########################################
#original_data训练的模型准确率
#数据切分
X_original = original_data.iloc[:, original_data.columns != 'Class']
y_original = original_data.iloc[:, original_data.columns == 'Class']
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=0)

#寻找最佳的正则化系数C
best_c_original = printing_Kfold_scores(X_train_original, y_train_original)
import time
start_original = time.time()
#正则化模型建立
lr_original = LogisticRegression(C=best_c_original, penalty='l1', solver='liblinear', class_weight='balanced')
#不经正则化模型
# lr_original = LogisticRegression()
lr_original.fit(X_train_original, y_train_original.values.ravel())
y_predict_original = lr_original.predict(X_test_original.values)
y_score_original = lr_original.score(X_test_original, y_test_original)

end_original = time.time()
print("original_data  L1 model training time:%.2fs"%(end_original-start_original))
print('original_data  L1 accuracy: ', y_score_original)

# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test_original, y_predict_original)
np.set_printoptions(precision=2)

# 绘制
class_names = [0, 1, 3]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

##########################################################
#################PCA模型
# PCA降维后的数据的训练模型，准确率
X_PCA = PCA_data.iloc[:, PCA_data.columns != 'Class']
y_PCA = PCA_data.iloc[:, PCA_data.columns == 'Class']
X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA, y_PCA, test_size=0.3, random_state=0)

#寻找最佳的正则化系数C
best_c_PCA = printing_Kfold_scores(X_train_PCA, y_train_PCA)

start_PCA = time.time()
#正则化模型建立
lr_PCA = LogisticRegression(C=best_c_PCA, penalty='l1', solver='liblinear', class_weight='balanced')
#不经正则化模型
# lr_PCA = LogisticRegression()
lr_PCA.fit(X_train_PCA, y_train_PCA.values.ravel())
y_predict_PCA = lr_PCA.predict(X_test_PCA.values)
y_score_PCA = lr_PCA.score(X_test_PCA, y_test_PCA)
end_PCA = time.time()
print("PCA_data L1 model training time:%.2fs"%(end_PCA-start_PCA))
print('PCA_data L1 accuracy: ', y_score_PCA)

# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test_PCA, y_predict_PCA)
np.set_printoptions(precision=2)

# 绘制
class_names = [0, 1, 3]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()



###################################################################
###################################################################
###################################################################
#后面是过采样模型，本次报告并没有写进去，供后续的学习
# from imblearn.over_sampling import SMOTE
#
# #原始数据过采样模型
# original_data_features = original_data.iloc[:, original_data.columns != 'Class']
# original_data_labels = original_data.iloc[:, original_data.columns == 'Class']
#
# #按照最多的类的数目进行过采样
# oversample_original = SMOTE(random_state=0)
# os_features_original, os_labels_original = oversample_original.fit_resample(original_data_features, original_data_labels)
#
# print(len(os_labels_original[os_labels_original==1]))
# print(os_labels_original)
# os_features_original = pd.DataFrame(os_features_original)
# os_labels_original = pd.DataFrame(os_labels_original)
# features_train_original, features_test_original, labels_train_original, labels_test_original = train_test_split(os_features_original, os_labels_original, test_size=0.2, random_state=0)
#
# best_c_original_over = printing_Kfold_scores(features_train_original, labels_train_original)
# #原始数据过采样模型训练,准确率：0.8503589177250138
# original_lr = LogisticRegression(C=best_c_original_over, penalty='l2')
# original_lr.fit(features_train_original, labels_train_original.values.ravel())
# y_predicts_original = original_lr.predict(features_test_original.values)
# y_scores_original = original_lr.score(features_test_original,labels_test_original)
# # 计算混淆矩阵
# cnf_matrix = confusion_matrix(labels_test_original,y_predicts_original)
# np.set_printoptions(precision=2)
#
# print("召回率: ", cnf_matrix[1, 1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
#
# # 绘制
# class_names = [0, 1, 3]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
# plt.show()
# print(y_scores_original)



# #PCA降维数据过采样模型
# PCA_data_features = PCA_data.iloc[:, PCA_data.columns != 'Class']
# PCA_data_labels = PCA_data.iloc[:, PCA_data.columns == 'Class']
#
# #按照最多的类的数目进行过采样
# oversample_PCA = SMOTE(random_state=0)
# os_features_PCA, os_labels_PCA = oversample_PCA.fit_resample(PCA_data_features, PCA_data_labels)
#
# print(len(os_labels_PCA))
# print(os_labels_PCA)
# os_features_PCA = pd.DataFrame(os_features_PCA)
# os_labels_PCA = pd.DataFrame(os_labels_PCA)
# features_train_PCA, features_test_PCA, labels_train_PCA, labels_test_PCA = train_test_split(os_features_PCA, os_labels_PCA, test_size=0.2, random_state=0)
#
# best_c_PCA_over = printing_Kfold_scores(features_train_PCA, labels_train_PCA)
# #原始数据过采样模型训练,0.3准确率：0.7730535615681944
# PCA_lr = LogisticRegression(C=best_c_PCA_over, penalty='l2')
# PCA_lr.fit(features_train_PCA, labels_train_PCA.values.ravel())
# y_predicts_PCA = PCA_lr.predict(features_test_PCA.values)
# y_scores_PCA = PCA_lr.score(features_test_PCA, labels_test_PCA)
# # 计算混淆矩阵
# cnf_matrix = confusion_matrix(labels_test_PCA, y_predicts_PCA)
# np.set_printoptions(precision=2)
#
# print("召回率: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
#
# # 绘制
# class_names = [0, 1, 3]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
# plt.show()
# print(y_score_original, y_score_PCA, y_scores_original, y_scores_PCA)