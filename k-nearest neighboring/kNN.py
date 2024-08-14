# coding=utf-8
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    # 计算距离（欧氏距离）
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet   # tile(array, shape) 重复array成shape形状
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1) # 按列求和，最后是一个列向量
    distances = sqDistances ** 0.5

    # 选择距离最小的k个点
    sortedDistIndicies = distances.argsort() # 默认从小到大排序，返回的是索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # dict.get(key, default) 返回key对应的value，如果key不存在则返回default
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # dict.items() 返回dict的key-value对，operator.itemgetter(1) 表示按照第二个元素排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() # 去掉首尾空格
        listFromLine = line.split('\t') # 以tab键分割
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1])) # 最后一列是类别
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0) # 按列求最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1)) # oldvlaue - min
    normDataSet = normDataSet / tile(ranges, (m, 1)) # newvalue = (oldvalue - min) / max-min
    return normDataSet, ranges, minVals

def datingClassTest(hoRatio=0.10, k=3):
    # hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio) # 测试集大小
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],\
                                     datingLabels[numTestVecs:m], k) # 选取前10%作为测试集
        print("the classifier came back with: %d, the real answer is: %d" \
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32): # 32*32的图片
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j]) # 一行一行读取，按列存储
    return returnVect

def handwritingClassTest(k=3):
    hwLabels = []
    trainingFileList = listdir('trainingDigits') # listdir() 返回指定目录下的文件名,例如['0_0.txt', '0_1.txt', ...]
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))  # 32*32=1024
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] # 去掉后缀
        classNumStr = int(fileStr.split('_')[0]) # 获取类别
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

def draw():
    import matplotlib
    import matplotlib.pyplot as plt
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels)) # scatter(x, y, size, color)
    #ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2]) # scatter(x, y)
    plt.xlabel('Frequent Flyier Miles Earned Per Year')
    plt.ylabel('Percentage of Time Spent Playing Video Games')
    plt.show()
