import json as js
import pandas as pd
import numpy as np
import os

class Shovel:

    def __init__(self, dataPath='VeReMi_training_set/'):
        self.splitCols=['pos','pos_noise','spd','spd_noise','acl','acl_noise','hed','hed_noise']
        self.dataPath=dataPath
        self.dirList=os.listdir(self.dataPath)
        self.dataPd=self.load_data()
        self.senderRoleDict,self.roleList=self.get_sender_role()
        self.dataByAttackDictPd=self.data_by_attack()

    def open_files(self):
        print('processing json files. '+str(len(self.dirList))+' files found.')
        messageID={}
        dataList=[]
        counter=0
        for fileName in self.dirList:
            with open(self.dataPath+fileName, "r") as fp:
                jsonFile=fp.read()
                jsonFile=jsonFile.replace("\n","")
                jsonFile='['+jsonFile
                jsonFile=jsonFile.replace("}","},")
                jsonFile=jsonFile[:-1]
                jsonFile=jsonFile+']'
                jsonFile=js.loads(jsonFile)
                for d in jsonFile:
                    if 'messageID' in d:
                        if not d['messageID'] in messageID:
                            messageID[d['messageID']]=1
                            dataList.append(d)
            counter+=1
            if counter%int(len(self.dirList)/5)==0:
                print("{:.2f}% files read.".format(100*counter/len(self.dirList)))
        dataPd=pd.DataFrame(dataList)
        print('splitting columns.')
        for col in self.splitCols:
            splitColPd=pd.DataFrame(dataPd[col].to_list(),columns=[col+'_x',col+'_y',col+'_z'])
            dataPd=pd.concat([dataPd,splitColPd],axis=1)
            dataPd=dataPd.drop(col,axis=1)
        print('saving processed files.')
        try:
            os.mkdir('processed_'+self.dataPath[:-1]+'/')
        except FileExistsError:
            print('directory processed_'+self.dataPath[:-1]+'/ found.')
        except Exception as e:
            print(e)
            quit()
        dataPd.to_csv('processed_'+self.dataPath[:-1]+'/processed_'+self.dataPath[:-1]+'.csv')
        return dataPd

    def load_data(self):
        print('getting data.')
        try:
            dataPd=(pd.read_csv('processed_'+self.dataPath[:-1]+'/processed_'+self.dataPath[:-1]+'.csv')).iloc[: , 1:]
            print('data loaded from file.')
            return dataPd
        except:
            try:
                dataPd=self.open_files()
                return dataPd
            except Exception as e:
                print(e)
                quit()

    def get_sender_role(self):
        dirList=self.dirList
        senderRoleDict={}
        roleList=[]
        for groundTruth in dirList:
            groundTruth=groundTruth.replace('.json','')
            groundTruth=groundTruth.split('-')
            senderRoleDict[float(groundTruth[1])]=groundTruth[2]
            if not groundTruth[2] in roleList:
                roleList.append(groundTruth[2])
        roleList.sort()
        return senderRoleDict,roleList

    def data_by_attack(self):
        try:
            roleDictList={}
            for role in self.roleList:
                roleDictList[role]=(pd.read_csv('processed_'+self.dataPath[:-1]+'/'+role+'_'+self.dataPath[:-1]+'.csv')).iloc[: , 1:]
            print('sorted data loaded from files.')
            return roleDictList

        except:
            roleDictList={}
            for index,row in self.dataPd.iterrows():
                if self.senderRoleDict[row['sender']] in roleDictList:
                    roleDictList[self.senderRoleDict[row['sender']]].append(row)
                else:
                    roleDictList[self.senderRoleDict[row['sender']]]=[]
                    roleDictList[self.senderRoleDict[row['sender']]].append(row)
                if index%int(self.dataPd.shape[0]/5)==0:
                    print("{:.2f}% entries sorted.".format(100*index/self.dataPd.shape[0]))
            print('saving sorted data.')
            try:
                os.mkdir('processed_'+self.dataPath[:-1]+'/')
            except FileExistsError:
                print('directory processed_'+self.dataPath[:-1]+'/ found.')
            except Exception as e:
                print(e)
                quit()
            for role in self.roleList:
                roleDictList[role]=pd.DataFrame(roleDictList[role])
                roleDictList[role].to_csv('processed_'+self.dataPath[:-1]+'/'+role+'_'+self.dataPath[:-1]+'.csv')
            return roleDictList