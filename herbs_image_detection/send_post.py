import urllib.request
import json
import os
import ssl
from PIL import Image
import io

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
test_path = os.path.join('D:/code/git_Andy/sage_git/herbs_image_detection/score/test_pic/Chantui.jpg')
data = open(test_path, 'rb').read()
import base64
image =  Image.open(test_path).convert('RGB')
buffer = io.BytesIO()
image.save(buffer, "JPEG", quality=90)
data = base64.b64encode(buffer.getvalue())
image.close()

body = data

url = 'https://chinese-herbs-ml-iidnz.eastus2.inference.ml.azure.com/score'
# Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
api_key = 'YmUNahh5HLZLlDRmZP5NYd1Uj1GEytKC'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'herbs-detect-server' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result.decode("utf8", 'ignore'))
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))