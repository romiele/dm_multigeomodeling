# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:05:57 2024
    File for the generation of images from FM
@author: romie
"""
import os
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import MDS

workdir = 'C:/Users/rmiele/Work/dm_multigeomodeling/'
os.chdir(workdir)
from model import DiffusionModel
from data import FaciesSet
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict


def tabulate_events(dpath):
    list_files = os.listdir(dpath)
    
    summary_iterators = []
    for file in list_files:
        if "events" in file: summary_iterators.append(file)
    del list_files
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in summary_iterators]
    
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []
    
    train_loss = summary_iterators[0].Scalars(tags[0])
    epochs = summary_iterators[0].Scalars(tags[1])
    val_loss = summary_iterators[0].Scalars(tags[2])

    steps_to_epoch = np.array([epochs[i].step for i in range(len(epochs))])
    epochs_to_epoch = np.array([epochs[i].value for i in range(len(epochs))])
    
    steps_to_trainloss = np.array([train_loss[i].step for i in range(len(train_loss))])
    loss_to_trainloss = np.array([train_loss[i].value for i in range(len(train_loss))])
    
    epochs_to_trainloss = []
    for i in range(len(steps_to_trainloss)):
        if len(np.where(steps_to_epoch==steps_to_trainloss[i])[0])>0:
            index = np.where(steps_to_epoch==steps_to_trainloss[i])
            epoch = int(epochs_to_epoch[index][0])
        epochs_to_trainloss.append(epoch)
            
    epochs_to_trainloss = np.array(epochs_to_trainloss)

    valloss_to_epoch = np.array([val_loss[i].value for i in range(len(val_loss))])
    
    train_loss = loss_to_trainloss
    epochs = epochs_to_trainloss
    val_loss = valloss_to_epoch
    
    avg_loss_e = np.zeros(len(val_loss))
    std_loss_e = np.zeros(len(val_loss))
    
    for i in range(len(val_loss)):
        avg_loss_e[i] = np.mean(train_loss[epochs_to_trainloss == i])
        std_loss_e[i] = np.std(train_loss[epochs_to_trainloss == i])
    
    return train_loss, epochs, val_loss, avg_loss_e, std_loss_e

def split_hist(data):
    data_f = data[:,0]
    data_ip = data[:,1]
    
    data_ip0 = data_ip[data_f<0.5]
    data_ip1= data_ip[data_f>0.5]
    
    return data_ip0, data_ip1

train_folder = '/lightning_logs_octopus/Ip_Facies_syn/28.10.24/'
workdir = 'C:/Users/rmiele/Work/dm_multigeomodeling/'
traindir = 'C:/Users/rmiele/Work/Training_Data/Synthetic_channels/'
img_size = 64
T_steps = 1000
img_depth = 2
n_samples = 1
chunks = 1

last_checkpoint = glob.glob(f"{workdir}/{train_folder}/checkpoints/*.ckpt")[-1]

train_loss, epochs, val_loss, avg_loss_e, std_loss_e = tabulate_events(f"{workdir}/{train_folder}/")

plt.figure()
plt.plot(np.arange(len(val_loss)),val_loss, label='Validation loss')
plt.errorbar(np.arange(len(avg_loss_e)), avg_loss_e, yerr=std_loss_e, capsize=2, 
             elinewidth=.5, fmt='--',label='Training loss')
plt.legend()
plt.savefig(f"{workdir}/{train_folder}/loss.png",dpi=400)
plt.show()

batch = int(n_samples/chunks)

model = DiffusionModel.load_from_checkpoint(
            last_checkpoint,
            in_size=img_size * img_size,
            t_range=T_steps,
            img_depth=img_depth, map_location="cuda:0"
        )


gen_samples = torch.zeros((n_samples,img_depth,img_size,img_size))
sampled_steps = []

sample_steps = torch.arange(model.t_range - 1, 0, -1).to('cuda')
sampled_t = 0

for X in range(chunks):
    x = torch.randn(
        (batch, img_depth, img_size, img_size)
    ).to('cuda')

# Denoise the initial noise for T steps
    for t in tqdm(sample_steps, desc="Sampling"):
        x = model.denoise_sample(x, t.to('cuda'))
    gen_samples[X*batch:(batch*X)+batch] = x

if img_depth>=2:
    fig, axs = plt.subplots(img_depth,1, figsize=(4,img_depth*4))
    for i in range(img_depth):
        axs[i].imshow(gen_samples[:,i].mean(0).squeeze().detach().cpu(), cmap='jet')
    plt.savefig(f"{workdir}/{train_folder}/DM_mean.png",dpi=400)

else: 
    plt.imshow(gen_samples.mean(0).squeeze().detach().cpu(), cmap='jet', vmin=0,vmax=1)
    plt.savefig(f"{workdir}/{train_folder}/DM_mean.png",dpi=400)

plt.show()
        
# %%
    
s_to_s = np.zeros((n_samples,2))

for xx in range(n_samples):
    sample = gen_samples[xx,0]
    sample[sample<0.5]=0
    sample[sample>=0.5]=1
    vals, counts = np.unique(sample, return_counts=True)
    s_to_s[xx] = counts
    
ratio = s_to_s[:,1]/s_to_s[:,0]


NTI= 500
facies_dataset = FaciesSet('', root_dir=traindir, multi=True if img_depth==2 else False)
TIs = torch.zeros(NTI,img_depth,64,64)

for i in range (TIs.shape[0]):
    TIs[i] = facies_dataset[i]

s_to_s_TI = np.zeros((n_samples,2))

for xx in range(n_samples):
    sample = TIs[xx,0]
    vals, counts = np.unique(sample, return_counts=True)
    s_to_s_TI[xx] = counts
    
ratio_TI = s_to_s_TI[:,1]/s_to_s_TI[:,0]


plt.hist(ratio_TI,  color='gray', bins= np.linspace(0,1,25), 
         label='TI',alpha=1, weights= np.ones(len(ratio_TI))/len(ratio_TI))
plt.hist(ratio,  color='r', bins= np.linspace(0,1,25), 
         label='DM',alpha=0.65, weights= np.ones(len(ratio))/len(ratio))
plt.legend()
plt.savefig(f"{workdir}/{train_folder}/ratio.png",dpi=400)
plt.show()

# %%
temp_0_DM, temp_1_DM = split_hist(gen_samples)
temp_0_TI, temp_1_TI = split_hist(TIs)


plt.figure()
plt.hist(temp_0_DM, color='gray', alpha=0.8, label='Ip Shale (DM)', bins=np.linspace(0,1,30))
plt.hist(temp_1_DM, color='orange', alpha=0.8, label='Ip Sand (DM)', bins=np.linspace(0,1,30))
plt.legend()
plt.savefig(f"{workdir}/{train_folder}/DM_iphist.png",dpi=400)
plt.show()

plt.figure()
plt.hist(temp_0_TI, color='gray', alpha=0.8, label='Ip Shale (TI)', bins=np.linspace(0,1,30))
plt.hist(temp_1_TI, color='orange', alpha=0.8, label='Ip Sand (TI)', bins=np.linspace(0,1,30))
plt.legend()
plt.savefig(f"{workdir}/{train_folder}/DM_iphist.png",dpi=400)
plt.show()


# %%

if n_samples>16:
    for im in range(img_depth):
        fig, axs = plt.subplots(4,4, figsize=(10,10), sharey=True, sharex=True)
        kk = 0
        for i in range(4):
            for j in range(4):
                k = j if i<1 else int(j/2)
                axs[i,j].imshow(gen_samples[kk,0])
                kk+=1
        
        plt.savefig(f"{workdir}/{train_folder}/DM_realizations_{im}.png",dpi=400)
    
        plt.show()


plt.hist(gen_samples[:,0].flatten().detach().cpu(), bins=30)
plt.show()






plt.imshow(gen_samples[:,0].mean(0).squeeze().detach().cpu(), cmap='jet', vmin=0.1, vmax=0.35)
plt.colorbar()
plt.savefig(f"{workdir}/{train_folder}/DM_f_mean.png",dpi=400)
plt.show()

plt.imshow(TIs.mean(0).squeeze(), cmap='jet', vmin=0.1, vmax=0.35)
plt.colorbar()
plt.savefig(f"{workdir}/{train_folder}/TI_mean.png",dpi=400)
plt.show()


plt.imshow(gen_samples[:,1].mean(0).squeeze().detach().cpu(), cmap='jet', vmin=0.1, vmax=0.35)
plt.colorbar()
plt.savefig(f"{workdir}/{train_folder}/DM_Ip_mean.png",dpi=400)
plt.show()



plt.imshow(gen_samples.std(0).squeeze().detach().cpu(), cmap='jet', vmin=0.3, vmax=0.5)
plt.colorbar()
plt.savefig(f"{workdir}/{train_folder}/DM_std.png",dpi=400)
plt.show()

plt.imshow(TIs.std(0).squeeze(), cmap='jet', vmin=0.3, vmax=0.5)
plt.colorbar()
plt.savefig(f"{workdir}/{train_folder}/TI_std.png",dpi=400)
plt.show()

fig, axs = plt.subplots(4,4, figsize=(10,10), sharey=True, sharex=True)
kk = 0
for i in range(4):
    for j in range(4):
        k = j if i<1 else int(j/2)
        axs[i,j].imshow(TIs[kk,0])
        kk+=1

plt.savefig(f"{workdir}/{train_folder}/TI_realizations.png",dpi=400)

plt.show()


# %% MDS results realizations
import skimage
mds = MDS(n_components=2,n_jobs=-1,  dissimilarity='precomputed', verbose=1)

alldata = np.concatenate((TIs,gen_samples), axis=0).squeeze()
alldata[alldata<0]=0

dissimil= np.full((alldata.shape[0],alldata.shape[0]), fill_value= 0)
for i in range(alldata.shape[0]):
    for j in range(i,alldata.shape[0]):
        dissimil[i,j] = skimage.metrics.hausdorff_distance(
            alldata[i],alldata[j], method='modified')
    print (i)

# scipy.spatial.distance.directed_hausdorff
dissimil_v = np.rot90(np.fliplr(dissimil))
dissimil = dissimil+dissimil_v

Tot_mds = mds.fit(dissimil)

# %%    
NTI = 500
fig, axs = plt.subplots(1,1, figsize=(4, 4), dpi=400)
axs.scatter(Tot_mds.embedding_[:NTI,0],Tot_mds.embedding_[:NTI,1], 
            zorder = 1, color='gray', alpha=0.9, label='Training data', s=30)

axs.scatter(Tot_mds.embedding_[NTI:,0],Tot_mds.embedding_[NTI:,1], 
            zorder = 2, color='darkred', alpha=0.8, marker= 'o', 
            label='Diffusion Model', s=30)

plt.legend()
plt.savefig(f"{workdir}/{train_folder}/MDS.png",dpi=400)
plt.show()
