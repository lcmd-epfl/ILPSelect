import numpy as np 
import copy

data=np.load("data.npz", allow_pickle=True)
data=dict(data)
size_database=len(data['database_labels'])

for target_index in range(len(data['target_labels'])): 
        CT=data['target_ncharges'][target_index]
        T=data['target_CMs'][target_index]
        I=[]
        for i in range(len(CT)):
            if CT[i] == 1:
                I.append(i)
        CT=np.delete(CT, I)
        T=np.delete(T, I, axis=0)
        T=np.delete(T,I,axis=1)
        data['target_ncharges'][target_index]=CT
        data['target_CMs'][target_index]=T

for i in range(size_database):
    print("    ", i)
    M=data['database_CMs'][i]
    CM=data['database_ncharges'][i]
    m=len(CM)
    J=[]
    for j in range(m):
        if CM[j]==1:
            J.append(j)
    CM=np.delete(CM,J)
    M=np.delete(M,J,axis=0)
    M=np.delete(M,J,axis=1)
    data['database_CMs'][i]=M
    data['database_ncharges'][i]=CM

labels='database_labels'
ncharges='database_ncharges'
CMs='database_CMs'

np.savez("qm7_CM_data.npz", vitc_qm7_labels=data[labels], vitc_qm7_ncharges=data[ncharges], vitc_qm7_CMs=data[CMs], vitd_qm7_labels=data[labels], vitd_qm7_ncharges=data[ncharges], vitd_qm7_CMs=data[CMs], qm9_qm7_labels=data[labels], qm9_qm7_ncharges=data[ncharges], qm9_qm7_CMs=data[CMs])
