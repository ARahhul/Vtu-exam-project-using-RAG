import urllib.request
import json
req = urllib.request.Request('http://localhost:8000/query', data=b'{"question": "test"}', headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
except Exception as e:
    print("STATUS", e.code)
    print("BODY:", e.read().decode())
