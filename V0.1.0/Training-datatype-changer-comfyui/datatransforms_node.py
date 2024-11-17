import torch
from safetensors.torch import save_file, load_file
from torchvision import transforms
from PIL import Image

class Img2Tensor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"image": ("IMAGE", ),
                             "trigger": ("INT", {"default":1})
                            }
                }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "img2tensor"
    CATEGORY = "training/datatype"

    @torch.inference_mode(mode=False)
    def img2tensor(self, image, trigger):
        self.trigger=trigger
        
        tensor= image
        tensor=tensor.clone().requires_grad_()
        return (tensor,)
    
class Tensor2Img:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"imagetensor": ("TENSOR", ),
                            }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "tensor2img"
    CATEGORY = "training/datatype"

    @torch.inference_mode(mode=False)
    def tensor2img(self, imagetensor):
        
        image=imagetensor
        return (image,)


NODE_CLASS_MAPPINGS={
    "Img2Tensor": Img2Tensor,
    "Tensor2Img":Tensor2Img,
}
NODE_DISPLAY_NAME_MAPPINGS ={
    "Img2Tensor":"Img2Tensor",
    "Tensor2Img":"Tensor2Img",
}