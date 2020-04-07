import torch

def set_parameter_requires_grad(model, feature_extract):
    print("Finetuning?:"+str(not feature_extract))
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
def verifiParametersToTrain(model, feature_extract):
  params_to_update = model.parameters()
  print("Parameters to learn...")
  if feature_extract:
      params_to_update = []
      for name,param in model.named_parameters():
          if param.requires_grad == True:
              params_to_update.append(param)
              print("\t",name)
  else:
      for name,param in model.named_parameters():
          if param.requires_grad == True:
              print("\t",name)
  return params_to_update

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']