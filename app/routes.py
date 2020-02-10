from app import application
from flask import jsonify, request
import torch, torchvision, os, sys, requests, io
from urllib.request import urlretrieve
from PIL import Image

@application.route('/')
@application.route('/index')
def index():
    torch.backends.quantized.engine = 'qnnpack'
    #model = load_model(None, model_urls['backbone'])
    print(torch.backends.quantized.supported_engines)
    backbone = load_model(None, model_urls['backbone_qnnpack'])
    return "Hello, World!"

@application.route('/predict', methods=['GET', 'POST'])
def predict():

    #if request.method == 'POST':
    if 'file' not in request.files:
        return jsonify({'error':'No source img file'})
    print('request_files: ', request.files)
    file = request.files.get('file')
    if not file:
        return jsonify({'error':'Not correct source img file'})

    image_bytes = file.read()
    #print('file: ', img_bytes)

    #try:
    img, orig_size = transform_image(image_bytes,
                                     mean=[0, 0, 0],
                                     std=[1, 1, 1])  # Dont normalize coco 2017 cause pytorch models uses normalization themself
    img = img.squeeze()
    # print('Memory in detection/utils/get_prediction before predicting and img_memory_size: ', memorycheck(img))
    torch.backends.quantized.engine = 'qnnpack'
    model = load_model(None, model_urls['model'])

    print('Model loaded. Object detection predicting...')
    model.eval()

    print('Processed image shape: ', img.size())
    #print('Original image size: ', orig_size)

    with torch.no_grad():
        prediction = model([img])[0]

    #except Exception as e:
    #    print('Can not predict! Internal error:', e)
    #    return jsonify({'error': 'Can not predict. Please try another image.'})

    #print(prediction)
    pred = {'error': ''}
    pred['prediction'] = prediction

    print('Done!')
    #print('prediction: ', pred['prediction'])
    return jsonify(pred)




def transform_image(image_bytes, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], device=torch.device('cpu')):
    #res = 320 if str(device) == 'cpu' else 640
    #print('Working on {}. Sizing to {}px'.format(device, res))
    my_transforms = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(res),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean,std)]
    )
    # print(io.BytesIO(image_bytes))
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_size = image.size
        #print(image)
        return my_transforms(image).unsqueeze(0), orig_size
    except Exception as e:
        print('Exception was raised while transforming image: ',e)
        return ''




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



