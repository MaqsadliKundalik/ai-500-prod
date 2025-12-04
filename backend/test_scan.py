"""
Test scan endpoint with dummy image
"""
import requests
import io
from PIL import Image

# Create a dummy pill image
img = Image.new('RGB', (224, 224), color='white')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Login
login_response = requests.post(
    "http://localhost:8001/api/v1/auth/login",
    data={
        "username": "test@example.com",
        "password": "password123"
    }
)
token = login_response.json()["access_token"]
print(f"✅ Logged in, token: {token[:20]}...")

# Upload scan
img_bytes.seek(0)  # Reset stream position
files = {'image': ('test_pill.jpg', img_bytes, 'image/jpeg')}
headers = {'Authorization': f'Bearer {token}'}

response = requests.post(
    "http://localhost:8001/api/v1/scans/image",
    files=files,
    headers=headers
)

print(f"\n📊 Response Status: {response.status_code}")
print(f"📊 Response:\n{response.json()}")
