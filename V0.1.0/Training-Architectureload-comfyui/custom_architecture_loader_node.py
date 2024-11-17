import torch

class Architecture_Linear:
    def __init__(self):
        self.linear=None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"input": ("TENSOR",),
                             "input_dim": ("INT", {"default":100, "min":1}),
                             "output_dim": ("INT", {"default":100, "min":1}),
                             "bias": ([True, False],),
                            }
                }

    RETURN_TYPES = ("TENSOR", "MODEL")
    RETURN_NAMES = ("output", "model")
    FUNCTION = "load_architecture"
    CATEGORY = "training/architecture"


    @torch.inference_mode(mode=False)
    def load_architecture(self, input, input_dim, output_dim, bias):
        if self.linear is None:
            self.linear=torch.nn.Linear(input_dim, output_dim, bias=bias)
            self.linear.train()
        output = self.linear(input)
        return (output, self.linear)
        
class Activation_SiLU:
    def __init__(self):
        self.SiLU=None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                            "input":("TENSOR", ),
                            "inplace": ([False, True],),
                            }
                }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "load_architecture"
    CATEGORY = "training/activation"


    @torch.inference_mode(mode=False)
    def load_architecture(self, input, inplace):
        if self.SiLU is None:
            self.SiLU=torch.nn.SiLU(inplace=inplace)
        output=self.SiLU(input)
        return (output, )
 
class Testnode_grad:
    def __init__(self):
        self.linear=torch.nn.Linear(100, 100)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "optional": {
                            "input":("INT", {"default":1}),
                            "inplace": ([False, True],),
                            }
                }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE=True
    FUNCTION = "testgrad"
    CATEGORY = "training"

    @torch.inference_mode(mode=False)
    def testgrad(self, input=None, inplace=None):

        print(torch.is_grad_enabled())
        #torch.no_grad(False)


        self.linear=torch.nn.Linear(5, 5)
        #torch.set_grad_enabled(True)
        input=torch.ones(5)#, requires_grad=True)
        print('input', input)
        #with torch.enable_grad():
        #with torch.inference_mode(False):
        print(torch.is_grad_enabled())
        output=self.linear(input)
        print(output.grad)
        print(output)
        return (output, )
    
NODE_CLASS_MAPPINGS={
    "Architecture_Linear": Architecture_Linear,
    "Activation_SiLU":Activation_SiLU,
    "Testnode_grad":Testnode_grad
}
NODE_DISPLAY_NAME_MAPPINGS ={
    "Architecture_Linear": "Architecture_Linear",
    "Activation_SiLU":"Activation_SiLU",
    "Testnode_grad":"Testnode_grad"
}