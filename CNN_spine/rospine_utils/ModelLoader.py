import torch


class ModelLoader(object):
    model = None

    def __init__(self, ckpt_path=None):
        self.model = torch.load(ckpt_path,map_location=torch.device('cpu'))

    def load_model(self, ckpt_path):
        self.model = torch.load(ckpt_path,map_location=torch.device('cpu'))

    def run_inference(self, input_data):
        if self.model is None:
            raise Exception("no valid model was loaded, can't perform inference")
        with torch.no_grad():
            output = self.model.forward(input_data)
        return output

    def to_device(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()
