
def set_parameter_requires_grad(model, freezeConvLayers):
    if freezeConvLayers:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True