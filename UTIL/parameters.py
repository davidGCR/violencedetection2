import torch

def set_parameter_requires_grad(model, freezeConvLayers):
    # print("Freezing all conv?:"+str(freezeConvLayers))
    if freezeConvLayers:
        for param in model.parameters():
            param.requires_grad = False
            
def verifiParametersToTrain(model, freezeConvLayers, printLayers=False):
    params_to_update = model.parameters()
    # print("Finetuning?:"+str(not feature_extract))
    # if freezeConvLayers:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            if printLayers:
                print("\t",name)
    # else:
    #     for name,param in model.named_parameters():
    #         if param.requires_grad == True:
    #             if printLayers:
    #                 print("\t",name)
    return params_to_update

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']