import torch
import constants
import torch.nn as nn
import numpy as np

def concatenate(x, numDiPerVideos, fn_encoder, fn_adapt):
    lista = []
    for dimage in range(0, numDiPerVideos):
        feature = fn_encoder(x[dimage])
        feature = fn_adapt(feature)
        feature = torch.flatten(feature, 1)
        # feature = feature.view(feature.size(0), self.num_ftrs)
        lista.append(feature)
    x = torch.cat(lista, dim=1)
    return x
    
def maxTemporalPool(x, numDiPerVideos, fn_encoder):
    # loss #torch.Size([8, 1, 3, 224, 224])
    # print('tempPool input: ', x.size())
    lista = []
    # seqLen = numDiPerVideos
    # print(seqLen)
    for dimage in range(0, numDiPerVideos):
        feature = fn_encoder(x[dimage])
        
        lista.append(feature)

    # resta = lista[0]-lista[1]
    # print('encoder out: ', torch.sum(resta))

    minibatch = torch.stack(lista, 0)
    minibatch = minibatch.permute(1, 0, 2, 3, 4)
    
    tmppool = nn.MaxPool2d((numDiPerVideos, 1))
    lista_minibatch = []
    for idx in range(minibatch.size()[0]):
        out = tempMaxPooling(minibatch[idx], tmppool)
        # print('tempMaxPooling output: ', out.size())
        lista_minibatch.append(out)

    feature = torch.stack(lista_minibatch, 0)
    # feature = torch.flatten(feature, 1)

    # print('tempPool output: ', feature.size())
    return feature

def tempMaxPooling(stacked_images, tmppool):
    # stacked_images:  torch.Size([1, 512, 7, 7])
    # out:  torch.Size([512, 7, 7])
    # stacked_images:  torch.Size([1, 512, 7, 7])
    # out:  torch.Size([512, 7, 7])
    # stacked_images:  torch.Size([1, 512, 7, 7])
    # out:  torch.Size([512, 7, 7])
    # stacked_images:  torch.Size([1, 512, 7, 7])
    # out:  torch.Size([512, 7, 7])
    spermute = stacked_images.permute(2,3,0,1)
    out = tmppool(spermute)
    out = out.permute(2, 3, 0, 1)
    out = torch.squeeze(out)
    # print('stacked_images: ', stacked_images.size())
    # print('out: ',out.size())
    return out

def tempMaxPooling2(stacked_images, tmppool):
    print('stacked_images: ',stacked_images.size())
    spermute = stacked_images.permute(2,3,0,1)
    out = tmppool(spermute)
    out = out.permute(2, 3, 0, 1)
    out = torch.squeeze(out)
    return out