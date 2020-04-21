import torch, requests, os
from torchvision.models.detection import maskrcnn_resnet50_fpn
from flask import _app_ctx_stack as stack
from flask import current_app
from copy import deepcopy

model_urls ={
    'model' : 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
    'model_qnnpack': '1-0Aq-T4LX2oZiFoCQjJgWfUWjO-cVRz9',
    'model_fbgemm': '1bj5-hLI3YPj2xPb5W_Vh1pJbhGHPnWvR'
}


class Model(object):

    def __init__(self, app=None):
        self.app = app
        if self.app is not None:
            self.init_app(app)

    def init_app(self, app):
        app.teardown_appcontext(self.teardown)

    def get_checkpoint(self, url, model_dir='/tmp/pretrained'):

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

        # Checking environment variable for Google bucket access
        # some .pth files are in my google drive, check is url a google drive token or URL
        google_drive = False if '/' in url else True  # url checking
        # print(f'os.path.exists(model_dir): {os.path.exists(model_dir)}\n')
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        filename = url.split('/')[-1]
        if google_drive: filename = url + '.pth'
        cached_file = os.path.join(model_dir, filename)
        # Checking isn't there a model weight file in local filesystem and load it if needed
        if os.path.exists(cached_file): return cached_file
        if not google_drive:
            print('Downloading: "{}" to {}\n'.format(url, cached_file))
            r = requests.get(url)
            with open(cached_file, 'wb') as f:
                f.write(r.content)
        else:
            print('Downloading: "{}" to {}\n'.format('model weights from google drive', cached_file))
            download_file_from_google_drive(url, cached_file)
        return cached_file

    def connect(self, model_name='model'):
        model_name =  'model_fbgemm'
        cached_file = self.get_checkpoint(model_urls[model_name])
        if model_name == 'model':
            model = maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
            model.load_state_dict(torch.load(cached_file))
        else:  # scripted model loading...
            model = torch.jit.load(cached_file)
        model.transform.max_size = 800
        model.transform.min_size = (640,)
        model.eval()

        return model

    def predict(self, x, opt=False):
        with torch.jit.optimized_execution(opt), torch.no_grad():
            out = self.model(x)
        return out

    def teardown(self, exception):
        ctx = stack.top
        if hasattr(ctx, 'torch_model'):
            del ctx.torch_model


    @property
    def model(self):
        ctx = stack.top
        if ctx is not None:
            if not hasattr(ctx, 'torch_model'):
                ctx.torch_model = self.connect()
            return ctx.torch_model