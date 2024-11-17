import torch
from safetensors.torch import save_file, load_file
from PIL import Image
import numpy as np

def bar_image(n, width=100, height=10):
    image_array = torch.zeros((1, height, width, 3), dtype=torch.uint8)
    
    image_array[:, :, :n] = torch.tensor([0, 255, 0], dtype=torch.uint8).view(1, 1, 3)
    
    image_array[:, :, n:] = torch.tensor([0, 0, 0], dtype=torch.uint8).view(1, 1, 3)
    
    return image_array

class CustomModelTrainer:
    def __init__(self):
        self.current_iter=1
        self.optims=None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                            "optims": ("OPTIMIZER",), #optimizer의 리스트
                            "loss": ("TENSOR",),
                            "epochs":("INT", {"default":5, "min":1}),
                            "iters":("INT", {"default":5, "min":1}),
                            "accumulate":("INT", {"default":1, "min":1,}),
                            }
                }

    RETURN_TYPES = ("MODEL", "STRING", "IMAGE")
    RETURN_NAMES = ("model", "logs", "progressbar")
    FUNCTION = "train_model"
    CATEGORY = "training"

    @torch.inference_mode(mode=False)
    def train_model(self, optims, loss, epochs, iters, accumulate):
        if self.optims is None:
            self.optims=optims
        total_iters=epochs*iters
        print(f"[{total_iters}/{self.current_iter}] iteration loss : {loss.item():.6f}")
        (loss/accumulate).backward()
        if self.current_iter%accumulate==0:
            for optim in self.optims:
                optim.step()
                optim.zero_grad()
        progress_bar=bar_image(n=int(self.current_iter/total_iters*100), width=100, height=3)
        self.current_iter+=1
        return ("None", "None", progress_bar)



class SetParameters:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                            "model": ("MODEL", )
                            }
                }

    RETURN_TYPES = ("PARAMETERS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "set_parameters"
    CATEGORY = "training"

    @torch.inference_mode(mode=False)
    def set_parameters(self, model):
        return (model.parameters(), )
    
class ConcatParameters:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
                "optional": {
                            "parameters1": ("PARAMETERS", ),
                            "parameters2": ("PARAMETERS", ),
                            "parameters3": ("PARAMETERS", ),
                            "parameters4": ("PARAMETERS", )
                            }
                }

    RETURN_TYPES = ("PARAMETERS",)
    RETURN_NAMES = ("parameter list", )
    FUNCTION = "concat_parameters"
    CATEGORY = "training"

    @torch.inference_mode(mode=False)
    def concat_parameters(self, parameters1=None,parameters2=None,parameters3=None,parameters4=None):
        parameters=[]
        if parameters1 is not None:
            parameters+=list(parameters1)
        if parameters2 is not None:
            parameters+=list(parameters2)
        if parameters3 is not None:
            parameters+=list(parameters3)
        if parameters4 is not None:
            parameters+=list(parameters4)
        
        return (parameters, )

class SavePrivateModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                            "model": ("MODEL", ),
                            "file_name":("STRING", {"default":"custom_model"}),
                            "file_type":(["totalmodel", "statedict", "safetensors"],)
                            }
                }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "save_model"
    CATEGORY = "training"

    def save_model(self, model, file_name, file_type):
        if file_type=='totalmodel':
            torch.save(model, file_name)
            return (model,)

class TrainInitializer:
    def __init__(self):
        self.ep=1
        self.iter=1
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                            "count":("INT", {"default":0}),
                            "epochs": ("INT", {"default":100, "min":1}),
                            "iters":("INT", {"default":100, "min":1}),
                            }
                }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("epochs", "iters",)
    FUNCTION = "counting_train"
    CATEGORY = "training"

    def counting_train(self, count, epochs, iters):
        self.max_ep=epochs
        self.max_iter=iters
        self.count=count
        if self.iter>self.max_iter:
            self.iter=1
            self.ep+=1
        if self.ep>self.max_ep:
            raise Exception(f"Train {self.max_ep} epochs has done.")
        self.iter+=1
        return (self.max_ep, self.max_iter,)

class PrintText:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"epochs": ("INT", {"default":100, "min":1}),
                             "iters": ("INT", {"default":100, "min":1}),
                            }
                }

    RETURN_TYPES = ()
    FUNCTION = "printnum"
    OUTPUT_NODE=True
    CATEGORY = "training"

    def printnum(self, epochs, iters):
        print(f"{epochs}, {iters}")
        return {}




NODE_CLASS_MAPPINGS={
    "CustomModelTrainer": CustomModelTrainer,
    "SavePrivateModel":SavePrivateModel,
    "SetParameters":SetParameters,
    "ConcatParameters":ConcatParameters,
    "PrintText":PrintText,
    "TrainInitializer":TrainInitializer,
}
NODE_DISPLAY_NAME_MAPPINGS ={
    "CustomModelTrainer": "CustomModelTrainer",
    "SavePrivateModel":"SavePrivateModel",
    "SetParameters":"SetParameters",
    "ConcatParameters":"ConcatParameters",
    "PrintText":"PrintText",
    "TrainInitializer":"TrainInitializer",
}