import requests
import os


def get_checkpoint(url, model_dir='/tmp/pretrained'):
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


