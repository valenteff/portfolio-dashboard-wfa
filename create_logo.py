from PIL import Image
from io import BytesIO
import base64
import requests

# Create a placeholder chart icon for the dashboard
url = 'https://cdn-icons-png.flaticon.com/512/1170/1170723.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img = img.resize((200, 200))
img.save('assets/chart-logo.png')
