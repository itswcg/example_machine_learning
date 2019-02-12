# -*- coding: utf-8 -*-
# @Author  : itswcg
# @File    : KNN
# @Time    : 19-2-11 上午11:56
# @Blog    : https://blog.itswcg.com
# @github  : https://github.com/itswcg

"""
KNN K近邻算法
1.先计算当前点到数据集中每个点的距离
2.按照距离递增排序
3.选取k个距离小的点
4.确定这k个点所在类别出现的频率
5.返回频率最高的类别作为预测分类
"""

import os
import operator
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def create_dataset():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    KNN
    :param inX: 用于分类的输入向量inX, 输入值
    :param dataSet: 样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return: 分类
    """
    dataSetSize = dataSet.shape[0]  # 获取测试级形状 (4，2)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile 平铺函数
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # axis 轴
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 排序， 返回索引
    classCount = {}  # 统计最近的点，出现的次数
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 次数多的，概率大
    return sortedClassCount[0][0]


class Appointment:
    def __init__(self, filename):
        self.filename = filename

    def file2matrix(self):
        with open(self.filename, 'r') as f:
            arrayLines = f.readlines()
            returnMat = zeros((len(arrayLines), 3))
            classLabelVecotor = []
            index = 0
            for line in arrayLines:
                line = line.strip()
                listFromLine = line.split('\t')
                returnMat[index, :] = listFromLine[0:3]
                classLabelVecotor.append(int(listFromLine[-1]))
                index += 1
            return returnMat, classLabelVecotor

    def autoNorm(self, dataSet):
        """归一化特征值
        newValue = (oldValue-min)/(max-min)
        """
        minVals = dataSet.min(0)  # 从列中选取最小值
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        # normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m, 1))
        normDataSet = normDataSet / tile(ranges, (m, 1))
        return normDataSet, ranges, minVals

    def classifyPerson(self):
        resultList = ['not at all', 'in small doses', 'in large doses']
        datingDataMat, datingLables = self.file2matrix()
        normMat, ranges, minVals = self.autoNorm(datingDataMat)
        inArr = array([])  # 输入值
        classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLables, 3)
        print('you will probably like this person {}'.format(resultList[classifierResult - 1]))


class Discern:

    def img2vector(self, filename):
        """图像处理为向量"""
        returnVect = zeros((1, 1024))
        with open(filename, 'r') as f:
            for i in range(32):
                lineStr = f.readline()
                for j in range(32):
                    returnVect[0, 32 * i + j] = int(lineStr[j])
            return returnVect

    def handwriting(self):
        hwLabels = []
        trainingFileList = os.listdir('../dataset/KNN/trainingDigits')
        m = len(trainingFileList)
        trainingMat = zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i, :] = self.img2vector('../dataset/KNN/trainingDigits/{}'.format(fileNameStr))

        testFileList = os.listdir('../dataset/KNN/testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = self.img2vector('../dataset/KNN/testDigits/{}'.format(fileNameStr))
            classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
            print('分类结果：{}， 真实结果：{}'.format(classifierResult, classNumStr))
            if classifierResult != classNumStr:
                errorCount += 1.0
        print('总共出错:{}'.format(errorCount))
        print('出错率:{}'.format(errorCount / float(mTest)))


if __name__ == '__main__':
    # group, labels = create_dataset()
    # print(classify0([0, 1], group, labels, 3))

    # appoint = Appointment('../dataset/KNN/datingTestSet.txt')
    # datingDataMat, datingLables = appoint.file2matrix()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLables), 15.0 * array(datingLables))
    # plt.show()
    # print(appoint.autoNorm(datingDataMat))
    d = Discern()
    d.handwriting()
