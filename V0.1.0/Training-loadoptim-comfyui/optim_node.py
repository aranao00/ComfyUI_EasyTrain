import torch
class Optimizer_AdamW:
    def __init__(self):
        pass
    #torch.optim.Adam(params, lr=0.001,
    #betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
    #amsgrad=False, *, foreach=None, maximize=False,
    #capturable=False, differentiable=False, fused=None)
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                            "parameters": ("PARAMETERS", ),
                            "learning_rate":("STRING", {"default":"0.00002"})
                            },
                }

    RETURN_TYPES = ("OPTIMIZER", )
    RETURN_NAMES = ("adamw", )
    FUNCTION = "load_optim"
    CATEGORY = "training/optimizers"

    @torch.inference_mode(mode=False)
    def load_optim(self, parameters, learning_rate):
        optim=torch.optim.AdamW(parameters, lr=float(learning_rate))
        return ([optim,],)

NODE_CLASS_MAPPINGS={
    "Optimizer_AdamW": Optimizer_AdamW
}
NODE_DISPLAY_NAME_MAPPINGS ={
    "Optimizer_AdamW": "Optimizer_AdamW"
}