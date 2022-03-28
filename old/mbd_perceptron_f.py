import time
start_time = time.time()
import json as js
import pandas as pd
import numpy as np
import os

def load_data(dataPath):
    dirList=os.listdir(dataPath)
    print('processing json files. '+str(len(dirList))+' files found.')
    messageID={}
    data=[]
    counter=0
    for fileName in dirList:
        with open(dataPath+fileName, "r") as fp:
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
                        data.append(d)
                        #df.at[index,'type']=d['type']
                        #df.at[index,'rcvTime']=d['rcvTime']
                        #df.at[index,'sendTime']=d['sendTime']
                        #df.at[index,'sender']=d['sender']
                        #df.at[index,'messageID']=d['messageID']
                        #counter=0
                        #for item in d['pos']:
                        #    df.at[index,'pos'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['pos_noise']:
                        #    df.at[index,'pos_noise'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['spd']:
                        #    df.at[index,'spd'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['spd_noise']:
                        #    df.at[index,'spd_noise'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['acl']:
                        #    df.at[index,'acl'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['acl_noise']:
                        #    df.at[index,'acl_noise'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['hed']:
                        #    df.at[index,'hed'+str(counter)]=item
                        #    counter+=1
                        #counter=0
                        #for item in d['hed_noise']:
                        #    df.at[index,'hed_noise'+str(counter)]=item
                        #    counter+=1
                        #number_list.remove(index)
                        #index+=1
        counter+=1
        if counter%1000==0:
            print("{:.2f}% files read.".format(100*counter/len(dirList)))
    data=pd.DataFrame(data)
    print('saving processed files.')
    data.to_csv('processed_'+dataPath[:-1]+'.csv')
    return data

dataPath='VeReMi_training_set/'

data=load_data(dataPath)

print(data)

####load training csv
#train_data = pd.read_csv('processed_training_data_0.csv') 
#train_response= pd.read_csv('training_key.csv')
#validate_response=pd.read_csv('test_template_key.csv')
#
#train_data=train_data.iloc[: , 1:]
#train_data=train_data.sort_values('messageID')
#train_data=train_data.join(train_response['prediction'])
#train_data=train_data.sort_index()
#train_response=train_data['prediction']
#train_prediction=train_data['prediction']
#train_messageID=train_data['messageID']
#train_sender=train_data['sender']
#train_data=train_data.drop(columns='type')
#train_data=train_data.drop(columns='rcvTime')
#train_data=train_data.drop(columns='sendTime')
#train_data=train_data.drop(columns='sender')
#train_data=train_data.drop(columns='messageID')
#train_data=train_data.drop(columns='prediction')
#train_data=train_data.drop(columns='pos2')
#train_data=train_data.drop(columns='pos_noise2')
#train_data=train_data.drop(columns='spd2')
#train_data=train_data.drop(columns='spd_noise2')
#train_data=train_data.drop(columns='acl2')
#train_data=train_data.drop(columns='acl_noise2')
#train_data=train_data.drop(columns='hed2')
#train_data=train_data.drop(columns='hed_noise2')
#
#validate_data = pd.read_csv('processed_test_data_0.csv') 
#validate_data=validate_data.iloc[: , 1:]
#validate_data=validate_data.sort_values('messageID')
#validate_data=validate_data.join(validate_response['prediction'])
#validate_data=validate_data.sort_index()
#validate_response=validate_data['prediction']
#validate_messageID=validate_data['messageID']
#
#validate_data=validate_data.drop(columns='type')
#validate_data=validate_data.drop(columns='rcvTime')
#validate_data=validate_data.drop(columns='sendTime')
#validate_data=validate_data.drop(columns='sender')
#validate_data=validate_data.drop(columns='messageID')
#validate_data=validate_data.drop(columns='pos2')
#validate_data=validate_data.drop(columns='pos_noise2')
#validate_data=validate_data.drop(columns='spd2')
#validate_data=validate_data.drop(columns='spd_noise2')
#validate_data=validate_data.drop(columns='acl2')
#validate_data=validate_data.drop(columns='acl_noise2')
#validate_data=validate_data.drop(columns='hed2')
#validate_data=validate_data.drop(columns='hed_noise2')
#validate_data=validate_data.drop(columns='prediction')
#
##normalize all remaining non-normalized features using z-score normalization
#for label, content in train_data.items():
#    #if 'Age'==label or 'Annual_Premium'==label or 'Vintage'==label:
#    train_data[label] = (train_data[label] - train_data[label].mean())/train_data[label].std()
#
#for label, content in validate_data.items():
#    #if 'Age'==label or 'Annual_Premium'==label or 'Vintage'==label:
#    validate_data[label] = (validate_data[label] - validate_data[label].mean())/validate_data[label].std()
#
#a=np.zeros((train_data.shape[0]))
#
#devide=10
#train_half={}
#train_half[0]=0
#test_half={}
#test_half[0]=0
#train_data1=train_data
#train_data_output=train_data
#train_response1=train_response
#validate_data1=validate_data
#current_validate=0
#train_K=np.ndarray
#for part in range(devide):
#    #prepare data and variables for training
#    max_iter=50
#    p=2
#    #train_response=pd.DataFrame
#    #train_response=train_data['Response']
#    #train_data=train_data.drop(columns='Response')
#    train_half[part+1]=int((train_data1.shape[0]/devide)*(part+1))
#    if part==0:
#        train_data=train_data1.iloc[train_half[part]:train_half[part+1],:]
#        train_sender=train_sender.iloc[train_half[part]:train_half[part+1]]
#        train_data_output=train_data
#        train_data=train_data.to_numpy(dtype='float32')
#        train_response=train_response1.iloc[train_half[part]:train_half[part+1]]
#        train_prediction=train_response
#        train_response=train_response.to_numpy(dtype='float32')
#        train_accuracy=np.zeros((max_iter,p))
#        #a=np.zeros((train_data1.shape[0]))
#        for i in range(train_response.shape[0]):
#            if (0==train_response[i]):
#                train_response[i]=-1
#
#    train_accuracy=np.zeros((max_iter,p))
#    #a=np.zeros((train_data1.shape[0]))
#    #prepare data and variables for validation
#    #validate_response=pd.DataFrame
#    #validate_response=validate_data['Response']
#    #validate_data=validate_data.drop(columns='Response')
#    test_half[part+1]=int((validate_data1.shape[0]/devide)*(part+1))
#    validate_data=validate_data1.iloc[test_half[part]:test_half[part+1],:]
#    validate_data=validate_data.to_numpy(dtype='float32')
#    #validate_response=validate_response.to_numpy()
#    #validate_accuracy=np.zeros((max_iter,p))
#    #validate_response=
#    validate_K=np.ndarray
#    #for i in range(validate_response.shape[0]):
#    #    if (0==validate_response[i]):
#    #        validate_response[i]=-1
#
#    #train
#    for k in range(p):
#        k+=1
#        #gram matrix
#        print(part)
#        part1=part+1
#        if part1==0:
#            print('test')
#        if part==0:
#            print('training matmul')
#            train_K=np.power((np.matmul(train_data,np.transpose(train_data),dtype='float32')),k+1)
#            print(train_K)    
#            a=np.zeros((train_data.shape[0]))
#        print('testing matmul')
#        validate_K=np.power((np.matmul(validate_data,np.transpose(train_data),dtype='float32')),k+1)
#        print(validate_K)
#        print('p: ',k+1)
#        if part==0:
#            for i in range(max_iter):
#                misses=0
#                for j in range(train_K.shape[0]):
#                    u=np.matmul((a*train_response),train_K[j],dtype='float32')
#                    if (0>=u*train_response[j]):
#                        a[j]=a[j]+1
#                        misses=misses+1
#                        train_prediction[j]=0
#                    else:
#                        train_prediction[j]=1
#                train_accuracy[i,k]=(train_K.shape[0]-misses)/train_K.shape[0]
#                print('iter: ', i)
#                print('train accuracy: ',train_accuracy[i,k])
#        for j in range(validate_K.shape[0]):
#            u=np.matmul((a*train_response),validate_K[j],dtype='float32')
#            if (u<0):
#                validate_response[current_validate]=0
#            else:
#                validate_response[current_validate]=1
#            current_validate+=1
#        break
#
#train_data_output.to_csv('train_data111.csv')
#train_response.to_csv('train_response111.csv')
#train_prediction.to_csv('train_pred111.csv')
#train_sender.to_csv('train_sender111.csv')


#validate_data=validate_data1.join(validate_response['prediction'])
#validate_data=validate_data.sort_values('messageID')
#validate_response=validate_data['messageID']
#validate_response=validate_response.join(validate_data['prediction'])
#validate_response.to_csv('test_resp_1.csv')
#print(validate_response)

print("--- %s seconds ---" % (time.time() - start_time))
