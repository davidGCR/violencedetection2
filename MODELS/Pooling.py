import torch
import constants
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    # loss #torch.Size([8, 2, 3, 224, 224])
    # print('tempPool input: ', x.size())
    lista = []
    # seqLen = numDiPerVideos
    # print(seqLen)
    for dimage in range(0, numDiPerVideos):
        feature = fn_encoder(x[dimage]) #torch.Size([8, 512, 7, 7])
        # print('fn_encoder out:', feature.size())
        
        lista.append(feature)

    # resta = lista[0]-lista[1]
    # print('encoder out: ', torch.sum(resta))

    minibatch = torch.stack(lista, 0) #torch.Size([2, 8, 512, 7, 7])
    # print('minibatch out:', minibatch.size())
    minibatch = minibatch.permute(1, 0, 2, 3, 4) #torch.Size([8, 2, 512, 7, 7])
    # print('minibatch permuted out:', minibatch.size())
    
    tmppool = nn.MaxPool2d((numDiPerVideos, 1))
    lista_minibatch = []
    for idx in range(minibatch.size()[0]):
        out = tempMaxPooling(minibatch[idx], tmppool) #torch.Size([512, 7, 7])
        lista_minibatch.append(out)

    feature = torch.stack(lista_minibatch, 0) #torch.Size([bs, 512,7,7])
    # print('feature output: ', feature.size())
    return feature

def tempMaxPooling(stacked_images, tmppool):
    # stacked_images:  torch.Size([ndi, 512, 7, 7])
    # out:  torch.Size([512, 7, 7])
    spermute = stacked_images.permute(2, 3, 0, 1)  #torch.size([7,7,2,512])
    
    out = tmppool(spermute) #torch.Size([7, 7, 1, 512])
    # print('tmppool output: ', out.size())
    out = out.permute(2, 3, 0, 1) #torch.Size([1, 512,7,7])
    out = torch.squeeze(out) #torch.Size([512,7,7])
    # print('stacked_images: ', stacked_images.size())
    # print('out: ',out.size())
    return out

def tempMaxPooling2(stacked_images, tmppool):
    # print('stacked_images: ',stacked_images.size())
    spermute = stacked_images.permute(2,3,0,1)
    out = tmppool(spermute)
    out = out.permute(2, 3, 0, 1)
    out = torch.squeeze(out)
    return out

def avgTemporalPool(x, numDiPerVideos, fn_encoder):
    lista = []
    for dimage in range(0, numDiPerVideos):
        feature = fn_encoder(x[dimage]) #torch.Size([8, 512, 7, 7])
        lista.append(feature)

    minibatch = torch.stack(lista, 0) #torch.Size([2, 8, 512, 7, 7])
    minibatch = minibatch.permute(1, 0, 2, 3, 4) #torch.Size([8, 2, 512, 7, 7])
    avgpool = nn.AvgPool2d((numDiPerVideos, 1))
    lista_minibatch = []
    for idx in range(minibatch.size()[0]):
        spermute = minibatch[idx].permute(2, 3, 0, 1)  #torch.size([7,7,2,512])
        out = avgpool(spermute) #torch.Size([7, 7, 1, 512])
        # print('tmppool output: ', out.size())
        out = out.permute(2, 3, 0, 1) #torch.Size([1, 512,7,7])
        out = torch.squeeze(out) #torch.Size([512,7,7])
        lista_minibatch.append(out)

    feature = torch.stack(lista_minibatch, 0) #torch.Size([bs,512,7,7])
    return feature

def stdTemporalPool(x, numDiPerVideos, fn_encoder):
    # x= torch.Size([ndi, bs, 3, 224, 224])
    m = avgTemporalPool(x, numDiPerVideos, fn_encoder)  #torch.Size([bs,512,7,7])
    std = torch.zeros(m.size())
    std = std.to(device)
    # std.cuda()
    for idx in range(0,numDiPerVideos):
        feature_map = fn_encoder(x[idx]) #torch.Size([bs, 512, 7, 7])
        std = std + torch.pow(torch.add(feature_map,-m),2)
    std = std/numDiPerVideos
    std = torch.sqrt(std)
    return std