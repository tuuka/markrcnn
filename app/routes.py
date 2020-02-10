from app import application
import torch, torchvision, os, sys, requests
from urllib.request import urlretrieve

@application.route('/')
@application.route('/index')
def index():
    #torch.backends.quantized.engine = 'qnnpack'
    #model = load_model(None, model_urls['backbone'])
    print(torch.backends.quantized.supported_engines)
    backbone = load_model(None, model_urls['backbone_qnnpack'])
    return "Hello, World!"




    return "Hello, World!"





model_urls ={
    'backbone' : '19MOic9ojbeMY0BGuZj1h20PWzr6YzoVU',
    'model' : '13TCXHl6evZpj0VN71DGFhAdLm2tz0eTs',
    'backbone_qnnpack': '1AaqKkSxcQ2OWbW6ZkNN35GugSJWi4Yu8'
}


def load_model(model, url, model_dir='/tmp/pretrained', map_location=torch.device('cpu')):
    # Checking environment variable for Google bucket access
    # some .pth files are in my google drive, check is url a google drive token or URL
    google_drive = False if '/' in url else True  # url checking
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    filename = url.split('/')[-1]
    if google_drive: filename = url + '.pth'
    cached_file = os.path.join(model_dir, filename)


    # Checking isn't there a model weight file in local filesystem and load it
    if os.path.exists(cached_file):
        print('File exists!')
        if model is not None:
            checkpoint = torch.load(cached_file, map_location=map_location)
            print('Model weights was loaded from cached_file: ', cached_file)
        else:    # scripted model loading...
            print('Scripted model loaded from local file! ')
            return torch.jit.load(cached_file, map_location=map_location)
    else:
        # trying to load from google cloud storage
        blob = None
        checkpoint = None
        #print('USE_GCS: ', USE_GCS)

        if checkpoint is None:
            # there is no google cloud storage info in config or no model weights file into it
            # downloading model weights to local storage
            if not google_drive:
                sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
                urlretrieve(url, cached_file)
            else:
                sys.stderr.write('Downloading: "{}" to {}\n'.format('model weights from google drive', cached_file))
                download_file_from_google_drive(url, cached_file)
            if model is not None:
                checkpoint = torch.load(cached_file, map_location=map_location)
            else:  # scripted model loading...
                print('Scripted model loading...')
                return torch.jit.load(cached_file, map_location=map_location)
                print('Memory in load_model after upload to storage and remove cachedfile: ', memorycheck())

    if 'model' in checkpoint.keys(): checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)

    del checkpoint
    return model


def download_file_from_google_drive(id, destination):
    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(resp, dest):
        CHUNK_SIZE = 32768
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torch.jit.annotations import Tuple, List, Dict


# Here we quntize/dequantize tensors only in BackboneWithFPN class

class IntermediateLayerGetterQuant(torchvision.models._utils.IntermediateLayerGetter):

    def __init__(self, model=torchvision.models.quantization.resnet50(pretrained=False, quantize=False,
                                                                      norm_layer=torch.nn.BatchNorm2d),
                 return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        super(IntermediateLayerGetterQuant, self).__init__(model, return_layers)

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == torchvision.models.quantization.resnet.QuantizableBottleneck or \
                    type(m) == torchvision.models.quantization.resnet.QuantizableBasicBlock:
                m.fuse_model()


class FeaturePyramidNetworkQuant(nn.Module):
    __constants__ = ['inner_blocks', 'layer_blocks']

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(FeaturePyramidNetworkQuant, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()
        self.FPN_add = nn.quantized.FloatFunctional()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        # if extra_blocks is not None:
        #    assert isinstance(extra_blocks, ExtraFPNBlock)
        # self.extra_blocks = extra_blocks

    def forward(self, x: Dict[str, torch.Tensor]):

        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # Assume that input quanted already, otherwise uncomment it
        # x = [self.quant(i) for i in x]

        # Some changes in original code for torchscript
        # cause torchscripit can only iterate through the ModuleList:
        in_results = []
        i = 0
        for mod in self.inner_blocks:
            in_results.append(mod(x[i]))
            i += 1
        l_results = [in_results[-1]]
        i = len(in_results) - 2
        while i >= 0:
            feat_shape = in_results[i].shape[-2:]
            inner_top_down = torch.nn.functional.interpolate(l_results[0], size=feat_shape, mode="nearest")
            l_results.insert(0, self.FPN_add.add(in_results[i], inner_top_down))
            i -= 1

        i = 0
        results = []
        for mod in self.layer_blocks:
            results.append(mod(l_results[i]))
            i += 1

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        # if self.extra_blocks is not None:
        out['pool'] = torch.nn.functional.max_pool2d(results[-1], 1, 2, 0)

        return out


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super().__init__()
        self.body = IntermediateLayerGetterQuant(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetworkQuant(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.out_channels = out_channels

    def forward(self, input):
        out = self.fpn(self.body(self.quant(input)))
        out = OrderedDict([(k, self.dequant(v)) for k, v in out.items()])
        return out

    def fuse_model(self):
        # torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == IntermediateLayerGetterQuant:
                m.fuse_model()


def resnet50_fpn_backbone(backbone=None, pretrained=False):
    if backbone is None:
        # backbone = torchvision.models.quantization.resnet50(pretrained=False, quantize=False, norm_layer=torchvision.ops.misc.FrozenBatchNorm2d)
        backbone = torchvision.models.quantization.resnet50(pretrained=False, quantize=False,
                                                            norm_layer=torch.nn.BatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)



