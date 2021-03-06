# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# please consider citing our paper if you use any part of the provided code
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------

import numpy as np
import os
import glob
import sys
import datetime,time
import glob
from pathlib import Path
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import gc
import copy
import re
from sklearn.neighbors import NearestNeighbors
import yaml
from STAVPR import LMDTW,alignDistance,cosineDistance,RandomProject

def loadData(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = re.split('\t|,',line.strip())
        curLine=[float(temp) for temp in curLine]
        data.append(curLine)
    return data

def writeData(data,fileName):
    fp1 = open(fileName, 'w')
    for i in range(len(data)):
        for j in range(len(data[0])):
            fp1.write(str(data[i][j])+'\t')
        fp1.write('\n')
    fp1.close()

def PRcurve_sequence(minSeqDisList,resultSeqID):
    global l,interval,tolerance
    pointNums=1000    # set number of P-R
    P = np.zeros(pointNums)    #list of precision
    R = np.zeros(pointNums)    #list of recall
    F=0.          # max F1-score
    for k in range(pointNums):
        TP=0
        FN=0
        FP=0
        for i in range(len(minSeqDisList)):
            if(minSeqDisList[i]<k*1.0/pointNums):   #compare the distance with a threshold
                if (i == 0):         #compensate for the first l/2 frame of first query seq.
                    TP = TP + l / 2  #depending on the alignment result of the first l/2 frame of first query seq.
                    FP = FP + 0      #e.g. if all are right, then TP=TP+l/2 FP=FP+0
                elif (i == aC_range[1] - l): #compensate for the last l/2 frame of last query seq.
                    TP = TP + l / 2  #depending on the alignment result of the last l/2 frame of last query seq.
                    FP = FP + 0      #e.g. if all are right, then TP=TP+l/2 FP=FP+0
                else:
                    midGT = int(i * interval + 0.5 * (l - 1))   #the mid ID of the query seq (also is the ground truth ID)
                    midRes = int((resultSeqID[i][0] + resultSeqID[i][1]) / 2)    #the mid ID of the result seq
                    if (abs(midGT - midRes) <= tolerance):
                        TP+=1
                    else:
                        FP+=1
            else:
                if (i == 0 or i == aC_range[1] - l):   #compensate for the first(last) l/2 frame of first(last) query seq.
                    FN = FN + l / 2     # when all query have true matched result, all negetive(N) result are false(FN)
                else:
                    FN+=1
        if(TP+FP!=0 and TP + FN!=0):
            P[k]=TP/(TP+FP)
            R[k] =TP/(TP + FN)
        elif(TP+FP==0):
            P[k] = 1
            R[k] = 0
        if (P[k] + R[k] != 0 and 2 * P[k] * R[k] / (P[k] + R[k]) > F):
            F = 2 * P[k] * R[k] / (P[k] + R[k])
    #     fp1 = open("./PR/pWSU71.txt", 'a')
    #     fp1.write(str(P[k]) + '\t')
    #     fp2 = open("./PR/rWSU71.txt", 'a')
    #     fp2.write(str(R[k]) + '\t')
    # fp1.write('\n')
    # fp1.close()
    # fp2.write('\n')
    # fp2.close()

    print("F-score:",F)
    plt.plot(R,P,"r")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=0, xmax=1.)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color="b",linestyle="--")
    ax=plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    plt.savefig("SPSU.png")

def PRcurve_singleImg(minDistanList, resultImgID):
    global tolerance
    pointNums=1000      # set number of P-R
    P = np.zeros(pointNums)   #list of precision
    R = np.zeros(pointNums)   #list of recall
    F=0.      # max F1-score
    for k in range(pointNums):
        TP = 0
        FN = 0
        FP = 0
        for i in range(len(minDistanList)):
            if (minDistanList[i] < k * 1.0 / pointNums):  #compare the distance with a threshold
                if (abs(i-resultImgID[i])<=tolerance):
                    TP += 1
                else:
                    FP += 1
            else:
                FN += 1
        if (TP + FP != 0 and TP + FN != 0):
            P[k] = TP / (TP + FP)
            R[k] = TP / (TP + FN)
        elif (TP + FP == 0):
            P[k] = 1
            R[k] = 0
        if (P[k] + R[k] != 0 and 2 * P[k] * R[k] / (P[k] + R[k]) > F):
            F = 2 * P[k] * R[k] / (P[k] + R[k])
    #     fp1 = open("./PR/pWSU71.txt", 'a')
    #     fp1.write(str(P[k]) + '\t')
    #     fp2 = open("./PR/rWSU71.txt", 'a')
    #     fp2.write(str(R[k]) + '\t')
    # fp1.write('\n')
    # fp1.close()
    # fp2.write('\n')
    # fp2.close()

    print("F-score of single image matching:", F)
    plt.plot(R,P,"r")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=0, xmax=1.)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color="b",linestyle="--")
    ax=plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    #plt.savefig("SPSU.png")
    plt.show()
    
def write_match_output(predictions, q_dir, r_dir, ds_name, K):
    outfile = 'STA-VPR_' + ds_name + '_predictions.txt'
    print('Writing matching results to', outfile)
    with open(outfile, 'w') as kap_out:
        kap_out.write('# model: STA-VPR \n')
        kap_out.write(f'# Top {K} results \n')
        for q_idx in range(len(predictions)):
            q_path = q_dir[q_idx]
            for r_idx in range(K):
                r_path = r_dir[predictions[q_idx, r_idx]]
                kap_out.write(q_path + ', ' + r_path + '\n')
                
def get_positives(q_coord, db_coord, thr):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(db_coord)

        distances, positives = knn.radius_neighbors(q_coord, radius=thr)

        return positives

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    figfile=open("STAVPRconfig.yaml","r",encoding="utf-8").read()
    config=yaml.load(figfile, Loader=yaml.FullLoader)
    ds_name = config["dataset"]
    # the file path of historical (H_path) reference data and all current (aC_path) query data
    if ds_name == 'vlcmucd':
        root_dir = config["H_path"]
        H_path = glob.glob(os.path.join(root_dir, "*/RGB/1_*"))
        aC_path = glob.glob(os.path.join(root_dir, "*/RGB/2_*"))
    elif ds_name in ['changesim', 'tsunami', 'gsv']:
        H_path = glob.glob(os.path.join(config["H_path"], '*'))
        aC_path = glob.glob(os.path.join(config["aC_path"], '*'))
        
    aC_range = [0, len(aC_path)]
    K = config["K"]
    tolerance = config["tolerance"]
    modelName = config["modelName"]   #vgg or densenet (the name of CNN model for extracting feature)
    GRPandRA_flag=config["GRPandRA_flag"]  #choose whether to do Gaussian Random Projection and Restricted Alignment
    singleImg_flag=config["singleImg_flag"]  #choose whether to print the result of single image matching
    seq_flag = config["seq_flag"]

    if (modelName=="vgg"):
        from vgg import vggFeature
        H, aC = vggFeature(H_path, aC_path)
    elif (modelName=="densenet"):
        from densenet import densenetFeature
        H, aC = densenetFeature(H_path, aC_path)
    print("extract feature finished!\n")

    if GRPandRA_flag:
        H,aC=RandomProject(H,aC)
        print("reduce dimension finished!\n")

    # compute distance matrix D_aCH (all current qurey data vs. historical reference data)
    #D_aCH=cosineDistance(aC,H)
    D_aCH = alignDistance(aC, H, GRPandRA_flag)
    print("size of distance matrix D_aCH:", np.shape(D_aCH))
    print("compute distance matrix D_aCH finished!\n")
    writeData(D_aCH,'test.txt')
    predictions = np.argsort(D_aCH, axis=1).astype(int)
    write_match_output(predictions, aC_path, H_path, ds_name, K)
    
    print('====> Calculating recall @ N')
    if ds_name in ['changesim', 'tsunami', 'gsv']:
        q_coord = [(int(Path(q_p).stem), 0) for q_p in aC_path]
        db_coord = [(int(Path(db_p).stem), 0) for db_p in H_path]
        numQ = len(aC_path)
        thr = 2
    elif ds_name == 'vlcmucd':
        q_coord = [(5 * int(Path(q_p).parents[1].parts[-1]), int(Path(q_p).stem[2:])) for q_p in aC_path]
        db_coord = [(5 * int(Path(db_p).parents[1].parts[-1]), int(Path(db_p).stem[2:])) for db_p in H_path]
        numQ = len(aC_path)
        thr = 2
    
    n_values = [1,5,10,20]
    correct_at_n = np.zeros(len(n_values))
    gt = get_positives(q_coord, db_coord, thr)

    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break

    recall_at_n = correct_at_n / numQ
    np.savetxt('stavpr_' + ds_name +'_recalls.txt', recall_at_n)
    print('Done calculating recall @ N')
    
    # D_aCH = loadData('test.txt')
    D_aCH = np.array(D_aCH)

    # code for single image matching
    if singleImg_flag:
        argminDistanList = np.argmin(D_aCH, axis=1)
        minDistanList = np.min(D_aCH, axis=1)
        PRcurve_singleImg(minDistanList, argminDistanList)

    # code for sequence matching (i.e. LM-DTW)
    if seq_flag:
        [aC_len, H_len]=np.shape(D_aCH)
        interval = 1  # the frame interval between two query sequences
        l = 20  # length of current qurey sequence
        count = int((aC_len-l)/interval+1) # count of match times, i.e. there are 'count' qurey sequences in total
        minSeqDisList= np.zeros(count)     # list for recording min sequence distance (i-th element is the ditance between i-th qurey seq and its matched seq)
        resultSeqID = np.zeros([count,2])   # list for recording the matched result of each qurey sequence (record start ID and end ID of result seq)

        print("start to LM-DTW process (sequence matching)!")
        for Cindex in range(count):
            # slice from D_aCH to get D_CH.
            # D_CH is the distance matrix between a query seq (length=l=20) and the historical reference seq.
            D_CH = copy.deepcopy(D_aCH[Cindex * interval:Cindex * interval + l, :])
            # run LM-DTW process. search the matched seq of query seq.
            # get the seq distance between them. and the ID(start and end frame) of matched seq.
            minSeqDisList[Cindex],resultSeqID[Cindex][0],resultSeqID[Cindex][1]=LMDTW(D_CH)
            sequenceStart =int(resultSeqID[Cindex][0])
            sequenceEnd =int(resultSeqID[Cindex][1])
            print("current query sequence:","{:04d}".format(Cindex*interval),"~","{:04d}".format(Cindex*interval+l-1),\
                "(midpoint:"+"{:04d}".format(int(Cindex * interval + 0.5 * (l - 1)))+")",\
                "   result:", "{:04d}".format(sequenceStart),"~","{:04d}".format(sequenceEnd),\
                "(midpoint:"+"{:04d}".format(int((sequenceStart + sequenceEnd) / 2))+")",\
                "   min seq distance:","{0:7.4f}".format(minSeqDisList[Cindex]))
        print("LM-DTW process (sequence matching) finished!")

        PRcurve_sequence(minSeqDisList,resultSeqID)
