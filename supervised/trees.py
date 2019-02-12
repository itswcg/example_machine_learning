# -*- coding: utf-8 -*-
# @Author  : itswcg
# @File    : trees.py
# @Time    : 19-2-12 上午10:50
# @Blog    : https://blog.itswcg.com
# @github  : https://github.com/itswcg

"""
决策树
划分数据集的大原则：将无序的数据变得更加有序
熵是信息的期望
"""

from math import log


def calShannonEnt(dataset):
    """计算熵， 有个公式"""
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 划分数据集的特征
    :param value: 特征的返回值
    :return:
    """
    retDataSet = []
    for feat in dataSet:
        if feat[axis] == value:
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis + 1:])
            retDataSet.append(reducedFeat)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """选择最优特征，即熵最大的"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList = [_[i] for _ in dataSet]
        uniqueValue = set(featList)
        newEntropy = 0.0
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet, i, value)
            print(subDataSet)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # print(calShannonEnt(dataSet))
    # print(splitDataSet(dataSet, 0, 0))
    print(chooseBestFeatureToSplit(dataSet))
