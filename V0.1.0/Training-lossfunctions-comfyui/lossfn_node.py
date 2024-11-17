import torch
class LossFN_MSE:
    def __init__(self):
        self.lossfn=None
    #size_average=None, reduce=None, reduction='mean'
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                            {"input":("TENSOR",),
                             "label":("TENSOR",),
                            "size_average":([True, False, None], {"default": None}),
                            "reduce":([True, False, None],{"default": None}),
                            "reduction":(["mean", "none", "sum"],)
                            }
                }

    RETURN_TYPES = ("TENSOR", )
    RETURN_NAMES = ("loss", )
    FUNCTION = "load_lossfn"
    CATEGORY = "training/loss_functions"


    @torch.inference_mode(mode=False)
    def load_lossfn(self, input, label, size_average, reduce, reduction):
        if self.lossfn is None:
            self.lossfn=torch.nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)
        loss=self.lossfn(input, label)
        return (loss,)

NODE_CLASS_MAPPINGS={
    "LossFN_MSE": LossFN_MSE
}
NODE_DISPLAY_NAME_MAPPINGS ={
    "LossFN_MSE": "LossFN_MSE"
}