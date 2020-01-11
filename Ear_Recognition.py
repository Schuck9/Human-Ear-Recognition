"""
A simple implementation of classifier for ear recognition
@data: 2019.12.22
@author: Tingyu Mo
"""
import pandas as pd
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score
from PCA import PCA_SVD
from KNN import KNN



class Ear_Classifier():
    def __init__(self,):
        self.img_shape = None
        self.img_size = (128,128)
    
    def data_loader(self,data_path,label_transform = None):
        class_name_list = os.listdir(data_path)
        img_list = []
        label_list = []
        for class_name in class_name_list:
            class_dir = os.path.join(data_path,class_name)
            img_name_list = os.listdir(class_dir)
            for img_name in img_name_list:
                img_path = os.path.join(class_dir,img_name)
                img = self.read_img(img_path)
                if img.shape == None:
                    img.shape = img.shape
                img = img.flatten()
                img_list.append(img)
                # label_name = int(class_name[1:])
                label_name = class_name
                label_list.append(label_name)
        return np.array(img_list),np.array(label_list)
    
    def dataset_split(self,train_data,train_target,test_size = 0.3,random_state=None):
        '''
        split datasets to training set and test set with predefined size
        '''
        # x_train,x_test, y_train, y_test = train_test_split(train_data,train_target,test_size=test_size, random_state=random_state)#划分数据集
        seq = np.array([range(len(train_data))][0])
        train_index = np.argwhere((seq+1)%4 != 0)#利用标签获取该类的索引
        test_index = np.argwhere((seq+1)%4 == 0)
        x_train = self.data_reshape(train_data[train_index])#利用该类的索引获取数据 
        y_train = train_target[train_index]
        x_test = self.data_reshape(train_data[test_index])
        y_test = train_target[test_index]
        return x_train,x_test, y_train, y_test

    def data_reshape(self,data):
        data_shape = data.shape
        if data_shape[1] == 1:
            data =data.reshape(data_shape[1],data_shape[0],data_shape[2])[0]
        return data

    def to_categorical(self,y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def read_img(self,img_path):
        Img = Image.open(img_path)
        Img = Img.resize(self.img_size)
        # img.show()    # 展示图片
        # print(im.size)   # 输出图片大小
        Img = np.asarray(Img)
        return Img

    def img_show(self,img_set):
        for img in img_set:
            img = np.real(img)
            img_size = img.size
            img_width = int(math.sqrt(img_size))
            img = img.reshape(img_width,img_width)
            img = Image.fromarray(img)
            img.show()
    
    def feature_face(self,components_mat):
        components_mat = components_mat.T
        dim,feature_face_num = components_mat.shape
        feature_face_num = 10
        for i in range(feature_face_num):
            feature_face = np.real(components_mat[:,i])*255*255*255
            feature_face = feature_face.reshape(self.img_size)
            feature_face = Image.fromarray(feature_face)
            # feature_face.show()
            feature_face = feature_face.convert('RGB')
            feature_face.save(os.path.join(Root_dir,"results/feature_ear/fe_{}.jpg".format(i)), quality=95)
    
    # def PCA_decomposition(self,data,n_components=2):
    #     print("PCA decomposition start!")
    #     pca = PCA(n_components=n_components)
    #     compress_data = pca.fit_transform(data)
    #     print("number of components: ",pca.n_components)
    #     print("explained variance ratio: ",pca.explained_variance_ratio_)
    #     return compress_data
 



if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion/Exp5-10'
    datasets_dir = os.path.join(Root_dir,"datasets")
    os.chdir(Root_dir)
    dataset_path = os.path.join(datasets_dir,"att_ear")
    EC = Ear_Classifier()
    PS = PCA_SVD()
    img,label = EC.data_loader(dataset_path)
    print("data loaded!")
    K_Nearest = KNN(kN = 3,method = "K_Nearest")
    # x_train,x_test,y_train,y_test = K_Nearest.dataset_split(img,label,0.2)
    x_train,x_test,y_train,y_test = EC.dataset_split(img,label,0.2)
    # _,pca = PS.PCA_decomposition_sklearn(x_train,n_components=60,svd_sovler = "full")
    # components_mat = pca.components_
    # x_train = pca.transform(x_train)
    # x_test = pca.transform(x_test)
    x_train,components_mat = PS.PCA_decomposition(x_train,n_components=10)
    x_test = PS.feature_mapping(x_test,components_mat)
    # EC.feature_face(components_mat)
    # EC.img_show(x_test)
    # x_test = PS.feature_mapping(x_test,np.real(components_mat))
    # EC.img_show(x_test)
    K_Nearest.train(x_train,y_train)
    y_pred = K_Nearest.predict(x_test)
    acc,prec = K_Nearest.evaluate(y_pred,y_test)
    print("acc: {} prec:{}".format(acc,prec))


    