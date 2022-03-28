import time
start_time = time.time()
import json as js
import pandas as pd
import numpy as np

#column_names=[
#        'type',
#        'rcvTime',
#        'sendTime',
#        'sender',
#        'messageID',
#        'pos0',
#        'pos1',
#        'pos2',
#        'pos_noise0',
#        'pos_noise1',
#        'pos_noise2',
#        'spd0',
#        'spd1',
#        'spd2',
#        'spd_noise0',
#        'spd_noise1',
#        'spd_noise2',
#        'acl0',
#        'acl1',
#        'acl2',
#        'acl_noise0',
#        'acl_noise1',
#        'acl_noise2',
#        'hed0',
#        'hed1',
#        'hed2',
#        'hed_noise0',
#        'hed_noise1',
#        'hed_noise2'
#    ]
#
#total_rows=701571
#number_list=[]
#for i in range(total_rows):
#    number_list.append(i)
#
#file="training_set.txt"
#directory='VeReMi_training_set/'
#
#with open(file, "r") as fp:
#    data_set_file_names = fp.readlines()
#
#process_data_set_file_names=[]
#for item in data_set_file_names:
#    item=item.replace("\n","")
#    process_data_set_file_names.append(item)
#
#index=0
#save_counter=0
#
#def open_data_set(index,save_counter):
#    df=pd.DataFrame(index=number_list,columns=column_names,dtype=float)
#    
#    save_progress=0 
#
#    for file_name in process_data_set_file_names:
#        with open(directory+file_name, "r") as fp:
#            data=fp.read()
#            data=data.replace("\n","")
#            data='['+data
#            data=data.replace("}","},")
#            data=data[:-1]
#            data=data+']'
#            data=js.loads(data)
#            for d in data:
#                if 'messageID' in d:
#                    df.at[index,'type']=d['type']
#                    df.at[index,'rcvTime']=d['rcvTime']
#                    df.at[index,'sendTime']=d['sendTime']
#                    df.at[index,'sender']=d['sender']
#                    df.at[index,'messageID']=d['messageID']
#                    counter=0
#                    for item in d['pos']:
#                        df.at[index,'pos'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['pos_noise']:
#                        df.at[index,'pos_noise'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['spd']:
#                        df.at[index,'spd'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['spd_noise']:
#                        df.at[index,'spd_noise'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['acl']:
#                        df.at[index,'acl'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['acl_noise']:
#                        df.at[index,'acl_noise'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['hed']:
#                        df.at[index,'hed'+str(counter)]=item
#                        counter+=1
#                    counter=0
#                    for item in d['hed_noise']:
#                        df.at[index,'hed_noise'+str(counter)]=item
#                        counter+=1
#                    number_list.remove(index)
#                    index+=1
#        process_data_set_file_names.remove(file_name)
#        save_progress+=1
#        print(save_progress)
#        print(file_name)
#        if save_progress==500:
#            print(index/701571.0)
#            df.to_csv('processed_training_data_'+str(save_counter)+'.csv')
#            save_counter+=1
#            return index,save_counter
#
##rows:701571
##columns: 29
#while(len(process_data_set_file_names)>0):
#    index,save_counter=open_data_set(index,save_counter)
#
##training_key=pd.read_csv('training_key.csv')
#
#

###load training csv
train_data = pd.read_csv('processed_training_data_0.csv') 
train_response= pd.read_csv('training_key.csv')
validate_response=pd.read_csv('test_template_key.csv')

train_data=train_data.iloc[: , 1:]
train_data=train_data.sort_values('messageID')
train_data=train_data.join(train_response['prediction'])
train_data=train_data.sort_index()
train_response=train_data['prediction']
train_messageID=train_data['messageID']

train_data=train_data.drop(columns='type')
train_data=train_data.drop(columns='rcvTime')
train_data=train_data.drop(columns='sendTime')
train_data=train_data.drop(columns='sender')
train_data=train_data.drop(columns='messageID')
train_data=train_data.drop(columns='prediction')
train_data=train_data.drop(columns='pos2')
train_data=train_data.drop(columns='pos_noise2')
train_data=train_data.drop(columns='spd2')
train_data=train_data.drop(columns='spd_noise2')
train_data=train_data.drop(columns='acl2')
train_data=train_data.drop(columns='acl_noise2')
train_data=train_data.drop(columns='hed2')
train_data=train_data.drop(columns='hed_noise2')

validate_data = pd.read_csv('processed_test_data_0.csv') 
validate_data=validate_data.iloc[: , 1:]
validate_data=validate_data.sort_values('messageID')
validate_data=validate_data.join(validate_response['prediction'])
validate_data=validate_data.sort_index()
validate_response=validate_data['prediction']
validate_messageID=validate_data['messageID']

validate_data=validate_data.drop(columns='type')
validate_data=validate_data.drop(columns='rcvTime')
validate_data=validate_data.drop(columns='sendTime')
validate_data=validate_data.drop(columns='sender')
validate_data=validate_data.drop(columns='messageID')
validate_data=validate_data.drop(columns='pos2')
validate_data=validate_data.drop(columns='pos_noise2')
validate_data=validate_data.drop(columns='spd2')
validate_data=validate_data.drop(columns='spd_noise2')
validate_data=validate_data.drop(columns='acl2')
validate_data=validate_data.drop(columns='acl_noise2')
validate_data=validate_data.drop(columns='hed2')
validate_data=validate_data.drop(columns='hed_noise2')
validate_data=validate_data.drop(columns='prediction')

#normalize all remaining non-normalized features using z-score normalization
for label, content in train_data.items():
    #if 'Age'==label or 'Annual_Premium'==label or 'Vintage'==label:
    train_data[label] = (train_data[label] - train_data[label].mean())/train_data[label].std()

for label, content in validate_data.items():
    #if 'Age'==label or 'Annual_Premium'==label or 'Vintage'==label:
    validate_data[label] = (validate_data[label] - validate_data[label].mean())/validate_data[label].std()

a=np.zeros((train_data.shape[0]))

devide=10
train_half={}
train_half[0]=0
test_half={}
test_half[0]=0
train_data1=train_data
train_response1=train_response
validate_data1=validate_data
current_validate=0
train_K=np.ndarray
for part in range(devide):
    #prepare data and variables for training
    max_iter=100
    p=3
    #train_response=pd.DataFrame
    #train_response=train_data['Response']
    #train_data=train_data.drop(columns='Response')
    train_half[part+1]=int((train_data1.shape[0]/devide)*(part+1))
    if part==0:
        train_data=train_data1.iloc[train_half[part]:train_half[part+1],:]
        train_data=train_data.to_numpy(dtype='float32')
        train_response=train_response1.iloc[train_half[part]:train_half[part+1]]
        train_response=train_response.to_numpy(dtype='float32')
        train_accuracy=np.zeros((max_iter,p))
        #a=np.zeros((train_data1.shape[0]))
        for i in range(train_response.shape[0]):
            if (0==train_response[i]):
                train_response[i]=-1

    train_accuracy=np.zeros((max_iter,p))
    #a=np.zeros((train_data1.shape[0]))
    #prepare data and variables for validation
    #validate_response=pd.DataFrame
    #validate_response=validate_data['Response']
    #validate_data=validate_data.drop(columns='Response')
    test_half[part+1]=int((validate_data1.shape[0]/devide)*(part+1))
    validate_data=validate_data1.iloc[test_half[part]:test_half[part+1],:]
    validate_data=validate_data.to_numpy(dtype='float32')
    #validate_response=validate_response.to_numpy()
    #validate_accuracy=np.zeros((max_iter,p))
    #validate_response=
    validate_K=np.ndarray
    #for i in range(validate_response.shape[0]):
    #    if (0==validate_response[i]):
    #        validate_response[i]=-1

    #train
    for k in range(p):
        k+=2
        #gram matrix
        print(part)
        part1=part+1
        if part1==0:
            print('test')
        if part==0:
            print('training matmul')
            train_K=np.power((np.matmul(train_data,np.transpose(train_data),dtype='float32')),k+1)
            print(train_K)    
            a=np.zeros((train_data.shape[0]))
        print('testing matmul')
        validate_K=np.power((np.matmul(validate_data,np.transpose(train_data),dtype='float32')),k+1)
        print(validate_K)
        print('p: ',k+1)
        if part==0:
            for i in range(max_iter):
                misses=0
                for j in range(train_K.shape[0]):
                    u=np.matmul((a*train_response),train_K[j],dtype='float32')
                    if (0>=u*train_response[j]):
                        a[j]=a[j]+1
                        misses=misses+1
                train_accuracy[i,k]=(train_K.shape[0]-misses)/train_K.shape[0]
                print('iter: ', i)
                print('train accuracy: ',train_accuracy[i,k])
        for j in range(validate_K.shape[0]):
            u=np.matmul((a*train_response),validate_K[j],dtype='float32')
            if (u<0):
                validate_response[current_validate]=0
            else:
                validate_response[current_validate]=1
            current_validate+=1
        break

validate_data=validate_data1.join(validate_response['prediction'])
validate_data=validate_data.sort_values('messageID')
validate_response=validate_data['messageID']
validate_response=validate_response.join(validate_data['prediction'])
validate_response.to_csv('test_resp_3.csv')
print(validate_response)

print("--- %s seconds ---" % (time.time() - start_time))
