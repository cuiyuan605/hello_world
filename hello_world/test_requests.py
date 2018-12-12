import requests as req
from io import BytesIO
import os
from PIL import Image

try:
    response = req.get("https://img.alicdn.com/tfscom/i4/392360086/O1CN011CVQVTbmHKZujAK_!!392360086.jpg_300x300.jpg",timeout=3)
    print(response.elapsed)
    print(response.elapsed.total_seconds())
    print(response.elapsed.microseconds)
    image_bytes=BytesIO(response.content)
    try:
        with Image.open(image_bytes) as img:
            print(1)
            image = img.convert('RGB')
    except Exception as e:
        print(2)
        print(e)
except req.exceptions.ConnectTimeout:
    print("1111")
except Exception as e:
    print("2222")