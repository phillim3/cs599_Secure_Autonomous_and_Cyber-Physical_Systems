import pandas as pd
import numpy as np

class Perceptron:

    def __init__(self,trainResponse,trainData,validateResponse,validateData,iterations=50,power=2):
        self.model=np.zeros(trainData.shape[0])
        self.trainAccuracy=np.zeros(iterations)
        self.validateAccuracy=np.zeros(iterations)
        self.trainResponse=trainResponse
        self.validateResponse=validateResponse
        self.iterations=iterations
        self.power=power
        self.trainK=self.gram_matrix(trainData,trainData,power)   
        self.validateK=self.gram_matrix(validateData,trainData,power)

    def gram_matrix(self,firstData,secondData,power):
        k=np.power((np.matmul(firstData,np.transpose(secondData))),power)
        return k

    def reset_perceptron(self,trainResponse,trainData,validateResponse,validateData,iterations=50,power=2):
        self.model=np.zeros(trainData.shape[0])
        self.trainAccuracy=np.zeros(iterations)
        self.validateAccuracy=np.zeros(iterations)
        self.trainResponse=trainResponse
        self.validateResponse=validateResponse
        self.iterations=iterations
        self.power=power
        self.trainK=self.gram_matrix(trainData,trainData,power)   
        self.validateK=self.gram_matrix(validateData,trainData,power)

    def train(self):
        for i in range(self.iterations):
            trainMisses=0
            for j in range(self.trainK.shape[0]):
                u=np.matmul((self.model*self.trainResponse),self.trainK[j])
                if (0>=u*self.trainResponse[j]):
                    self.model[j]=self.model[j]+1
                    trainMisses+=1
            self.trainAccuracy[i]=(self.trainK.shape[0]-trainMisses)/self.trainK.shape[0]
            print('iter: ', i)
            print('train accuracy: ',self.trainAccuracy[i])
            validateMisses=0
            for j in range(self.validateK.shape[0]):
                u=np.matmul((self.model*self.trainResponse),self.validateK[j])
                if (0>=u*self.validateResponse[j]):
                    validateMisses+=1
            self.validateAccuracy[i]=(self.validateK.shape[0]-validateMisses)/self.validateK.shape[0]
            print('validate accuracy: ',self.validateAccuracy[i])