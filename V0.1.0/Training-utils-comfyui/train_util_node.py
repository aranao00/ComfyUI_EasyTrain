import torch
class Util_View:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                            {"input":("TENSOR",),
                             "batchsize":("INT", {"default":1}),
                             "dim1":("INT", {"default":1}),
                             "dim2":("INT", {"default":1}),
                             "dim3":("INT", {"default":1}),
                             "dim4":("INT", {"default":1}),
                             "dim5":("INT", {"default":1}),
                             "dim6":("INT", {"default":1}),
                            }
                }

    RETURN_TYPES = ("TENSOR", )
    RETURN_NAMES = ("tensor", )
    FUNCTION = "util_view"
    CATEGORY = "training/utils"

    @torch.inference_mode(mode=False)
    def util_view(self, input, batchsize, dim1,dim2,dim3,dim4,dim5,dim6):
        if dim1==0:
            tensor=input.view(batchsize)
        elif dim2==0:
            tensor=input.view(batchsize, dim1)
        elif dim3==0:
            tensor=input.view(batchsize, dim1, dim2)
        elif dim4==0:
            tensor=input.view(batchsize, dim1, dim2,dim3)
        elif dim5==0:
            tensor=input.view(batchsize, dim1, dim2,dim3,dim4)
        elif dim6==0:
            tensor=input.view(batchsize, dim1, dim2,dim3,dim4,dim5)
        else:
            tensor=input.view(batchsize, dim1, dim2,dim3,dim4,dim5,dim6)
        return (tensor,)
class Util_Squeeze:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                            {"input":("TENSOR",),
                             "dim":("INT", {"default":1}),
                            }
                }

    RETURN_TYPES = ("TENSOR", )
    RETURN_NAMES = ("tensor", )
    FUNCTION = "util_squeeze"
    CATEGORY = "training/utils"

    @torch.inference_mode(mode=False)
    def util_squeeze(self, input, dim):
        tensor=torch.squeeze(input, dim)
        return (tensor,)
NODE_CLASS_MAPPINGS={
    "Util_View": Util_View,
    "Util_Squeeze":Util_Squeeze,
}
NODE_DISPLAY_NAME_MAPPINGS ={
    "Util_View": "Util_View",
    "Util_Squeeze":"Util_Squeeze",
}