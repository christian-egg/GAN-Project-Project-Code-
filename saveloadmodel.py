import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


"""
saves the given model to the given (string) path
Note that generator and discriminator will need to be saved to different files
"""
def save_model(path: str, model):
    torch.save(model.state_dict(), open(path, 'wb'))

"""
loads a model from the given (string) path; the argument "model" should be an empty initialized object extending from torch.nn.Module, and will be returned when the function is complete
"""
def load_model(path: str, model: nn.Module):
    a = torch.load(open(path, 'rb'))
    model.load_state_dict(a)
    return model

# example usage of the code
if __name__ == '__main__':
    models0 = __import__("models0")
    print(load_model("modelsave", models0.Generator()))