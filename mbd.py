import time
start_time = time.time()
import json as js
import pandas as pd
import numpy as np

def open_data_set(file,directory=""):
    column_names=[
        'type',
        'rcvTime',
        'sendTime',
        'sender',
        'messageID',
        'pos0',
        'pos1',
        'pos2',
        'pos_noise0',
        'pos_noise1',
        'pos_noise2',
        'spd0',
        'spd1',
        'spd2',
        'spd_noise0',
        'spd_noise1',
        'spd_noise2',
        'acl0',
        'acl1',
        'acl2',
        'acl_noise0',
        'acl_noise1',
        'acl_noise2',
        'hed0',
        'hed1',
        'hed2',
        'hed_noise0',
        'hed_noise1',
        'hed_noise2'
    ]
    total_rows=701571
    number_list=[]
    for i in range(total_rows):
        number_list.append(i)

    df=pd.DataFrame(index=number_list,columns=column_names,dtype=float)
    with open(file, "r") as fp:
        data_set_file_names = fp.readlines()

    index=0
    save_progress=0
    for file_name in data_set_file_names:
        file_name=file_name.replace("\n","")
        with open(directory+file_name, "r") as fp:
            data=fp.read()
            data=data.replace("\n","")
            data='['+data
            data=data.replace("}","},")
            data=data[:-1]
            data=data+']'
            data=js.loads(data)
            for d in data:
                if 'messageID' in d:
                    df.at[index,'type']=d['type']
                    df.at[index,'rcvTime']=d['rcvTime']
                    df.at[index,'sendTime']=d['sendTime']
                    df.at[index,'sender']=d['sender']
                    df.at[index,'messageID']=d['messageID']
                    counter=0
                    for item in d['pos']:
                        df.at[index,'pos'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['pos_noise']:
                        df.at[index,'pos_noise'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['spd']:
                        df.at[index,'spd'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['spd_noise']:
                        df.at[index,'spd_noise'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['acl']:
                        df.at[index,'acl'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['acl_noise']:
                        df.at[index,'acl_noise'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['hed']:
                        df.at[index,'hed'+str(counter)]=item
                        counter+=1
                    counter=0
                    for item in d['hed_noise']:
                        df.at[index,'hed_noise'+str(counter)]=item
                        counter+=1
                    index+=1
        save_progress+=1
        if save_progress==500:
            print(index/701571.0)
            df.to_csv('processed_test_data_1.csv')
            save_progress=0
    return df

#rows:701571
#columns: 29

df=open_data_set("training_set.txt",'VeReMi_training_set/')
df.to_csv('processed_training_data_1.csv')
#training_key=pd.read_csv('training_key.csv')
print(df)

##load training csv
#train_data = pd.read_csv('processed_training_data.csv') 
#
#train_data=train_data.drop(columns='type')
#train_data=train_data.drop(columns='rcvTime')
#train_data=train_data.drop(columns='sendTime')
#train_data=train_data.drop(columns='sender')
#train_data=train_data.drop(columns='messageID')
#
##normalize all remaining non-normalized features using z-score normalization
#for label, content in train_data.items():
#    if 'Age'==label or 'Annual_Premium'==label or 'Vintage'==label:
#        train_data[label] = (train_data[label] - train_data[label].mean())/train_data[label].std()
#
#prepare data and variables for training
#max_iter=100
#p=2
#train_response=pd.DataFrame
#train_response=train_data['Response']
#train_data=train_data.drop(columns='Response')
#train_data=train_data.to_numpy()
#train_response=train_response.to_numpy()
#train_accuracy=np.zeros((max_iter,p))
#a=np.zeros((train_data.shape[0]))
#train_K=np.ndarray
#for i in range(train_response.shape[0]):
#    if (0==train_response[i]):
#        train_response[i]=-1

#train
#for k in range(p):
#    #gram matrix
#    train_K=np.power((np.matmul(train_data,np.transpose(train_data))),k+1)
#    validate_K=np.power((np.matmul(validate_data,np.transpose(train_data))),k+1)
#    print('p: ',k+1)
#
#    for i in range(max_iter):
#        misses=0
#        for j in range(train_K.shape[0]):
#            u=np.matmul((a*train_response),train_K[j])
#            if (0>=u*train_response[j]):
#                a[j]=a[j]+1
#                misses=misses+1
#        train_accuracy[i,k]=(train_K.shape[0]-misses)/train_K.shape[0]
#        print('iter: ', i)
#        print('train accuracy: ',train_accuracy[i,k])
#
#        #validate
#        misses=0
#        for j in range(validate_K.shape[0]):
#            u=np.matmul((a*train_response),validate_K[j])
#            if (0>=u*validate_response[j]):
#                misses=misses+1
#        validate_accuracy[i,k]=(validate_K.shape[0]-misses)/validate_K.shape[0]
#        print('validate accuracy: ',validate_accuracy[i,k])

print("--- %s seconds ---" % (time.time() - start_time))