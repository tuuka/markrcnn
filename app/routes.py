from app import application

from flask import jsonify, request, render_template
import torch, os, sys, requests, io, random, colorsys, base64,time
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from urllib.request import urlretrieve
from PIL import Image
import psutil, json


labels = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush', 'hair brush',
]


model_urls ={
    'model' : 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
    'model_qnnpack': '1-0Aq-T4LX2oZiFoCQjJgWfUWjO-cVRz9'
}

def load_model(model, url, model_dir='/tmp/pretrained', map_location=torch.device('cpu')):

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

    if 'model' in checkpoint.keys(): checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)

    del checkpoint
    return model


def hex_colours(col):
    def clamp(x):
        return max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(col[0]), clamp(col[1]), clamp(col[2]))

# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return [hex_colours((round(r*255),round(g*255),round(b*255))) for r,g,b in colors]


torch.set_grad_enabled(False)
#print('Supported engines: ', torch.backends.quantized.supported_engines)
#torch._C._jit_set_profiling_executor(False)
#torch._C._jit_set_profiling_mode(False)
#torch.jit.optimized_execution(False)
torch.backends.quantized.engine = 'qnnpack'
model = load_model(None, model_urls['model_qnnpack'])
#model = maskrcnn_resnet50_fpn(pretrained=True)
model.transform.max_size = 800
model.transform.min_size = (640,)

model.eval()
# model warm-up
'''
t = time.time()
with torch.jit.optimized_execution(True), torch.no_grad():
    for i in range(1):
        model([torch.randn(3, 320, 480)])
dt = time.time() - t
print('Model warm-up time: %0.02f seconds\n' % dt)
'''

@application.route('/')
@application.route('/index')
def index():
    return render_template('index.html', title='MaskRCNN')


@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'No source img file'})
    t = time.time()
    file = request.files.get('file')
    if not file:
        return jsonify({'error':'Not correct source img file'})

    file = file.read()
    pred = prediction(file)

    data = json.dumps({'boxes': pred['boxes'],
                       'labels': pred['labels'].tolist(),
                       'scores': pred['scores'].tolist(),
                       'masks': pred['masks'],
                       'colors': pred['colors'],
                       'img_size': None,
                       'image_orig_size': pred['orig_size']
                       })

    dt = time.time() - t
    print('Model predict time: %0.02f seconds.' % dt)

    return jsonify({'error':'',
                    'data':data,
                    'time': round(dt),
                    'memory':str(psutil.Process(os.getpid()).memory_info().rss / 1024) + 'kiB',
                    'p_id':os.getpid()
                    })


def prediction(file):
    global model
    t = time.time()
    img = Image.open(io.BytesIO(file)).convert('RGB')
    orig_size = img.size
    img = transforms.ToTensor()(img)
    #print(img)
    #model.eval()
    print('Processed image shape: ', img.size())
    XYXY = list(img.size())[1:][::-1]
    XYXY = torch.tensor(XYXY + XYXY).float()
    with torch.jit.optimized_execution(False), torch.no_grad():
        prediction = model([img])

    # scripted model returns a tuple
    if type(prediction) is tuple:
        prediction = prediction[1]
    prediction = prediction[0]  # Only one image (first) is needed

    dt = time.time() - t

    pred = {
        'boxes'  : [],
        'labels' : [],
        'colors' : [],
        'scores': [],
        'time': {"all_time": round(dt)},
        'orig_size' : list(orig_size), # width first
    }
    N = len(prediction['scores'][prediction['scores'] > 0.7])
    pred['colors'] = random_colors(N)

    # Take TopN scores prediction and convert all to list cause
    # tensor is not JSON serializable
    for i in range(N):
        pred['boxes'].append((prediction['boxes'][i].cpu() / XYXY).tolist())
        pred['labels'].append(labels[prediction['labels'][i]])
        pred['scores'].append(prediction['scores'][i].item())

    # making color mask for instance segmentation by putting objects with high scores above objects with less scores
    if (prediction.get('masks') is not None) & (len(prediction['masks'])>0):
        #print("'prediction['masks']: ", prediction['masks'])
        r = torch.zeros_like(prediction['masks'][0]).byte()
        g = torch.zeros_like(prediction['masks'][0]).byte()
        b = torch.zeros_like(prediction['masks'][0]).byte()

        # sort in order of box areas
        idx = torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in pred['boxes']]).argsort(descending=True)

        for i in idx:
            mask = (prediction['masks'][i] > 0.5)
            # converting hex to rgb ..
            r[mask], g[mask], b[mask] = tuple(int(pred['colors'][i][j:j+2], 16) for j in (1, 3, 5))

        img_seg = torch.cat([r, g, b], 0)
        img_seg = Image.fromarray(img_seg.permute(1, 2, 0).byte().cpu().numpy()).resize(orig_size)
        buffered = io.BytesIO()
        img_seg.save(buffered, format="JPEG")
        pred['masks'] = 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")
        #del buffered, r, g, b, mask

    #del img, prediction, N
    #print('\npred:\n', pred)
    return pred