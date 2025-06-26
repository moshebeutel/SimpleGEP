from simplegep.models.resnet import resnet20

model_hub = {'resnet20': resnet20}

def get_model(model_name):
    assert model_name in model_hub, 'Model {} not found'.format(model_name)
    return model_hub[model_name]()