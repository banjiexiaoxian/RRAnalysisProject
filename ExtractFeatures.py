import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import UF as unionFind
from sklearn.neighbors import NearestNeighbors


# Method to read the File Containing the Locations and Time
def readTextFile(filename):
    rr = np.loadtxt(filename)
    rr = rr / 1000
    # rr = np.loadtxt('MIT-BIH/非正常人/男/中年人（30~50）/chf210.txt',usecols = (2),dtype = float,skiprows=1)
    # rr = np.loadtxt(filename, usecols=(2), dtype=float, skiprows=1)
    rr = np.array(rr)
    return rr

def cutSlice(rr,segment,segment_length):
    start = segment*segment_length
    end = start+segment_length
    rr_segment = rr[start:end]
    rr_segment = rr_segment[rr_segment < 2]
    rr_segment = rr_segment[rr_segment > 0.3]
    rr_i = rr_segment[0:len(rr_segment) - 1]
    rr_j = rr_segment[1:len(rr_segment)]
    elements = np.c_[rr_i, rr_j]
    elements = pd.DataFrame(elements)
    elements = elements.drop_duplicates()
    return elements.values
    # return elements
# Step 2: Construct SNN graph from the sparsified matrix---------------------------------------
def countIntersection(listi, listj):
    intersection = 0
    for i in listi:
        if i in listj:
            intersection = intersection + 1
    return intersection

# Method to compute SNN_similarity_matrix
def snn_sim_matrix(X,k):
    """
    利用sklearn包中的KDTree,计算节点的共享最近邻相似度(SNN)矩阵
    :param X: array-like, shape = [samples_size, features_size]
    :param k: positive integer(default = 5), 计算snn相似度的阈值k
    :return: snn距离矩阵
    """
    try:
        X = np.array(X)
    except:
        raise ValueError("输入的数据集必须为矩阵")
    samples_size, features_size = X.shape  # 数据集样本的个数和特征的维数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    ##由于有很多重复点的缘故，离的很近的点计算的KNN_matrix却不在里面，所以k需要加大，或者在计算时去掉重复点（在生成elemnents前就应先去掉重复点）
    knn_matrix = nbrs.kneighbors(X, return_distance=False)  # 记录每个样本的k个最近邻对应的索引
    sim_matrix = np.zeros((samples_size, samples_size))  # snn相似度矩阵
    for i in range(0, samples_size - 1):
        nextIndex = i + 1
        for j in range(nextIndex, samples_size):
            if j in knn_matrix[i] and i in knn_matrix[j]:
                count1 = countIntersection(knn_matrix[i], knn_matrix[j])
                sim_matrix[i][j] = count1
                sim_matrix[j][i] = count1
    return sim_matrix

# ---------end of findKNNList---------------------------------

def judgeLocation(point,line1_k,line1_b,line2_k,line2_b):
    if point[1]-line1_k*point[0]-line1_b == 0 or point[1]-line2_k*point[0]-line2_b == 0:
        return str('OnLine')
    elif point[1]-line1_k*point[0]-line1_b > 0:
        if point[1]-line2_k*point[0]-line2_b > 0:
            return 'SlowReduce'
        else:
            return 'FastReduce'
    else:
        if point[1]-line2_k*point[0]-line2_b > 0:
            return 'SlowPlus'
        else:
            return 'FastPlus'

def doSNN(elements,k,minPoints,eps,ith_figure,datasetname):
    features = dict()
    # Step 1: Preprocess dta and find K nearest neighbors-----------------------------------------
    similarityMatrix = snn_sim_matrix(elements, k)
    # np.save("similarityMatrix.npy",similarityMatrix)
    elements_index = [i for i in range(len(elements))]
    # Step2 filter corepoint----------------------------------------------------------------------

    def core(x, y):
        if x >= minPoints:
            return y
        else:
            return None

    def coreornot(x):
        if x >= minPoints:
            return True
        else:
            return False

    snnDensity1 = [None for i in range(len(similarityMatrix))]
    ##不能用 sum 来筛选核心点，这样与点的数量的关系太敏感了
    for i in range(len(similarityMatrix)):
        snnDensity1[i] = sum(similarityMatrix[i])
    corepoint = []
    for i in range(len(snnDensity1)):
        index = core(snnDensity1[i], i)
        if index != None:
            corepoint.append(index)
    # print(elements[corepoint, 0])
    ##Step3 画出所有点
    # plt.subplot(1,2,1)
    # plt.plot(elements[:, 0], elements[:, 1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=3,
    #          label='Not CorePoints')
    # plt.title('Corepts='+str(Corepts))
    # plt.xlabel('RR(i)(sec)',fontsize=15)
    # plt.ylabel('RR(i+1)(sec)',fontsize=15)
    # plt.xlim(0.3,2)
    # plt.ylim(0.3,2)
    # plt.show()
    # #Step4 画出核心点
    #
    # plt.plot(elements[corepoint,0],elements[corepoint,1],'o',markerfacecolor='y',markeredgecolor='y', markersize=3,
    #          label='CorePoints')
    # plt.title('corepoints with minpts='+str(minPoints) +' and maxpts='+str(maxPoints))
    # plt.xlim(0.3,1)
    # plt.ylim(0.3,1)
    # plt.legend(fontsize=15)
    # plt.show()
    corepoint_index = [i for i in range(len(corepoint))]
    corepoint = np.c_[corepoint, corepoint_index]
    # # step5 cluster corepoint，the cluster of corepoint means the cluster of whole elements
    union_find = unionFind.UF(corepoint)
    ##counstruct union tree from a 无向图
    epsMatrix = np.zeros((len(corepoint),len(corepoint)))
    for i in range(len(corepoint)):
        for j in range(len(corepoint)):
            if i != j :
                epsMatrix[i][j] = similarityMatrix[corepoint[i][0]][corepoint[j][0]]
    # np.save("epsMatrix.npy",epsMatrix)
    for i in corepoint:
        for j in corepoint:
            if i[0] != j[0] and similarityMatrix[i[0]][j[0]] >= eps:
                union_find.union(i[1], j[1])
    # corepoint分簇图
    clusters = union_find.clustered_elements()
    # print(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    cluster_no = 0
    # plt.subplot(1,2,2)
    plt.xlabel('RR(i)(sec)',fontsize=15)
    plt.ylabel('RR(i+1)(sec)',fontsize=15)
    plt.xlim(0.3,1.5)
    plt.ylim(0.3,1.5)

    ##将所有簇按点的个数排序
    clusters = dict(sorted(clusters.items(), key = lambda x: len(x[1]), reverse=True ))
    for cluster_index, element in clusters.items():
        plt.plot(elements[element, 0], elements[element, 1], 'o', markerfacecolor=colors[cluster_no],
                 markersize=8,label='Cluster'+str(cluster_no)+':'+str(len(element)))
        cluster_no += 1
    ###吸引子个数###
    features['cluster_no'] = cluster_no
    # Step1 选取主簇
    maxnum_cluster_index = list(clusters.keys())[0]
    # ##画出等速线和划分慢加速区和快加速区的线

    # Step2 计算主簇在等值线上的中心点
    major_element = clusters[maxnum_cluster_index]
    point_in_line = []
    for element in major_element:
        if abs(elements[element,0] - elements[element,1])<=0.01:
            point_in_line.append(elements[element,0])
    if len(point_in_line) == 0:
        features['isPassMiddleLine'] = False
    else:
        features['isPassMiddleLine'] = True
        point_in_line.sort()
        middle_point = point_in_line[int(len(point_in_line) / 2)]
        line2_k = -1
        line2_b = middle_point + middle_point
        reversed_equalSpeedLine_X = [middle_point + middle_point, 0]
        reversed_equalSpeedLine_Y = [0, middle_point + middle_point]
        plt.plot(reversed_equalSpeedLine_X, reversed_equalSpeedLine_Y, '-')
        line1_k = 1
        line1_b = 0
        #分别判断各个簇属于哪个位置区间
        cluster_location = dict()
        for cluster_index,element in clusters.items():
            if cluster_index == maxnum_cluster_index:
                continue
            locationInfo = dict()
            for point_index in element:
                point = elements[point_index]
                tmp_key = judgeLocation(point,line1_k,line1_b,line2_k,line2_b)
                if tmp_key in locationInfo.keys():
                    locationInfo[tmp_key] += 1
                else:
                    locationInfo[tmp_key] = 0
            locationInfo = sorted(locationInfo.items(),key = lambda x : x[1],reverse=True)
            cluster_location[cluster_index] = locationInfo[0][0]
        features['cluster_location'] = cluster_location
    equalSpeedLine_X = [0, 2]
    equalSpeedLine_Y = [0, 2]
    plt.plot(equalSpeedLine_X, equalSpeedLine_Y, '-')
    plt.xlim(0.3, 2)
    plt.ylim(0.3, 2)
    plt.legend(fontsize=15)
    plt.title('k=' + str(k) + ',Corepts=' + str(minPoints) + ',eps=16' + ',segment' + str(ith_figure))
    # plt.savefig(datasetname + str(ith_figure))
    # plt.cla()
    plt.show()
        # fig = plt.gcf()
        # fig.set_size_inches(10.5, 10.5)
        # fig.savefig('test2png.png', bbox_inches='tight', dpi=600)

    return features
# Main
if __name__ == "__main__":
    # k,minPoints,and eps are user defined inputs
    k = 28  # Nearest Neighbor Size
    ##点密度越大，Minpoints和eps就应该越大
    Corepts = 300# defines core point threshold
    # maxPoints = 10000
    eps = 15  # defines noise(点稀疏时，如赵博的数据，eps就得低，比如为8正好；点密集时，如MIT-BIH的数据，eps就得高，比如为20正好；如果可以自动调整eps，则是非常有意义的)
    # datasetpath = 'MIT-BIH/非正常人/男/老年人（50以上）/'
    # datasetname = 'chf204'
    datasetpath = 'SenseOn/'
    datasetname = '刘言灵17_11_29_13_2017_11_29_22_44'
    filename = datasetpath+datasetname+'.txt'
    rr = readTextFile(filename)
    # segment = 3
    segment_length = 3600
    rs = ''
    for segment in range(0, math.floor(len(rr) / segment_length) ):
        elements = cutSlice(rr, segment, segment_length)
        # np.save('chf205_格子.npy',elements)
        # print(len(elements))
        features = doSNN(elements,k,Corepts,eps,segment,datasetname)
        ##保存features
        rs = rs + 'segment' + str(segment) + ':'
        for feature_name,feature_val in features.items():
           rs = rs + feature_name + ':' + str(feature_val)+ '\t'
        rs = rs + '\n'
    fp = open(datasetname+"rs.txt",'w')
    fp.write(rs)



            #用于经验调参
            # print(snnDensity1)
            # X = range(0,len(snnDensity1))
            # Y = sorted(snnDensity1)
            # plt.plot(X,Y)
            # plt.show()