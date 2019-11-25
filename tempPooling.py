import torch
import torch.nn as nn
import numpy as np

# def tempMaxPooling(stacked_images, tmppool):
#     # stacked_images:  torch.Size([1, 512, 7, 7])
#     # out:  torch.Size([512, 7, 7])
#     # stacked_images:  torch.Size([1, 512, 7, 7])
#     # out:  torch.Size([512, 7, 7])
#     # stacked_images:  torch.Size([1, 512, 7, 7])
#     # out:  torch.Size([512, 7, 7])
#     # stacked_images:  torch.Size([1, 512, 7, 7])
#     # out:  torch.Size([512, 7, 7])
#     spermute = stacked_images.permute(2,3,0,1)
#     out = tmppool(spermute)
#     out = out.permute(2, 3, 0, 1)
#     out = torch.squeeze(out)
#     # print('stacked_images: ', stacked_images.size())
#     # print('out: ',out.size())
#     return out

# def tempMaxPooling2(stacked_images, tmppool):
#     print('stacked_images: ',stacked_images.size())
#     spermute = stacked_images.permute(2,3,0,1)
#     out = tmppool(spermute)
#     out = out.permute(2, 3, 0, 1)
#     out = torch.squeeze(out)
#     return out

# def main():
#     t1 = torch.tensor(torch.arange(1, 19))
#     t1 = torch.reshape(t1,(2,3,3))
#     t1 = t1*2.3
#     t1 = t1.float()
    
#     t2 = torch.tensor(torch.arange(19, 37))
#     t2 = torch.reshape(t2,(2,3,3))
#     # t2 = t2*0.3
#     t2 = t2.float()
    
#     t3 = torch.tensor(torch.arange(37, 55))
#     t3 = torch.reshape(t3,(2,3,3))
#     # t3 = t3*0.7
#     t3 = t3.float()
    
#     t4 = torch.tensor(torch.arange(55,73))
#     t4 = torch.reshape(t4,(2,3,3))
#     t4 = t4*0.5
#     t4 = t4.float()
#     lista = [t1,t2,t3,t4]
#     ldis = torch.stack(lista,0)
    
#     minibatch = torch.stack([0.3*ldis, 0.2*ldis, 0.0333*ldis, 0.04*ldis, 0.55*ldis],0)
#     # minibatch = minibatch.permute(1,0,2,3,4)

#     print('minibatch size: ', minibatch.size())
#     # print(minibatch)
    
#     torch.set_printoptions(precision=10,sci_mode=False)

#     # print('ldis size: ',ldis.size())
#     # print(ldis)
    
#     num_dynamic_images = minibatch.size()[1]
#     tmppool = nn.MaxPool2d((num_dynamic_images, 1))
#     for idx in range(minibatch.size()[0]):
#         out = tempMaxPooling(minibatch[idx], tmppool)
#         print('============================================')
#         print('video images: ',minibatch[idx])
#         print('out size: ',out.size())
#         print(out)

# if __name__ == "__main__":
#     main()
    