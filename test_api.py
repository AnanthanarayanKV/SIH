#test_api.py
import time
from fastapi.testclient import TestClient

from api import app, TTS_AVAILABLE

client = TestClient(app)

print("=" * 60)
print("HEALTH")
print("=" * 60)
r = client.get("/health")
print(r.status_code, r.json())

print("\n" + "=" * 60)
print("ROOT")
print("=" * 60)
r = client.get("/")
print(r.status_code, r.json())

print("\n" + "=" * 60)
print("TRANSLATE (valid batch)")
print("=" * 60)
r = client.post("/translate", json={"sentences": ["नमस्कार", "पानी", "माँ"]})
print(r.status_code)
print(r.json())

print("\n" + "=" * 60)
print("TRANSLATE (empty list)")
print("=" * 60)
r = client.post("/translate", json={"sentences": []})
print(r.status_code, r.json())

print("\n" + "=" * 60)
print("SPEAK (empty text -> 400 expected)")
print("=" * 60)
r = client.post("/speak", json={"text": ""})
print(r.status_code, r.json())

print("\n" + "=" * 60)
print("SPEAK (valid text)")
print("=" * 60)
r = client.post("/speak", json={"text": "ᱫᱟᱜ"})
print(r.status_code)
if r.status_code == 200:
    print("Content-Type:", r.headers.get("content-type"))
    print("Body length (bytes):", len(r.content))
else:
    print(r.json())

print("\nSPEAK available:", TTS_AVAILABLE)
