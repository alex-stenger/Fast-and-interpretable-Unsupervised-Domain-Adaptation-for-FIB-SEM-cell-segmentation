import torch
import tifffile
import torch.utils.data as data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .train import prepare_labels

## Merge les canals d'une image en one-hot à afficher
#def merge_canal(img) : 
#    tmp = np.zeros((img.shape[1], img.shape[2]))
#    for i in range(img.shape[0]) :
#        for k in range(img.shape[1]) :
#            for l in range(img.shape[2]) :
#                #print(img[i][k,l].shape)
#                if img[i][k,l].item() != 0 : 
#                    tmp[k,l] = img[i][k,l]
#    return(tmp)

#img of shape C*H*W
#def merge_channel(img) :
#    for i in range(img.shape[0]) :
#        img[img==0] = i
    
#    return(np.amax(img, axis=0))

#img du type C*H*W
def merge_canal(img) :
    tmp = np.zeros((img.shape[1],img.shape[2]))
    for i in range(img.shape[0]) :
        tmp[img[i]==1] = i 
    return(tmp)


def source_printer(dataloader_source, device) :
    real_batch = next(iter(dataloader_source))


    print("images source : ", real_batch[0].shape)
    print("mask source :", real_batch[1].shape)
    
    tmp = prepare_labels(real_batch[1], 19, device)
                       
    fig = plt.figure(figsize=(10,10)) # specifying the overall grid size
    fig.suptitle("training exemple source", fontsize=20)

    plt.subplot(2,1,1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(real_batch[0][0,0].cpu())
    plt.subplot(2,1,2)
    
    plt.imshow(tmp[0,1].cpu())
    print(np.max(tmp[0,1].flatten().cpu().numpy()))
    plt.show()
    
    test = merge_canal(tmp[0].cpu().numpy())
    plt.imshow(test, cmap = "tab20")
    
    #print(np.shape(tmp[0,1].flatten().cpu().numpy()))
    #cpt = 0
    #for i in tmp[0,1].flatten().cpu().numpy() :
    #    print(i)
    #    if i == 1 :
    #        cpt +=1
    #print(cpt)
    
    #print(real_batch["mask"][0])
    #plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    real_batch = next(iter(dataloader_source))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training images source - a batch")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    
def target_printer(dataloader_target, device) :
    real_batch = next(iter(dataloader_target))
    print("images target : ", real_batch["image"].shape)
    print("mask target :", real_batch["mask"].shape)
                       
    fig = plt.figure(figsize=(10,10)) # specifying the overall grid size
    fig.suptitle("training exemple target", fontsize=20)

    plt.subplot(2,1,1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(real_batch["image"][0,0].cpu(), cmap="gray")
    plt.subplot(2,1,2)
    plt.imshow(real_batch["mask"][0,0].cpu(), cmap="gray")
    #print(real_batch["mask"][0])
    #plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    real_batch = next(iter(dataloader_target))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training images target - a batch")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))