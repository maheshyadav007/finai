import time, hmac, hashlib, requests

# ---- Put your keys here ----
API_KEY    = "VJOGAFCUKX7S2XIX"
API_SECRET = "Rz1miB9WrT4AOxYDAaxbAaJe2eCZAj"
BASE_URL   = "https://api.india.delta.exchange"
# ----------------------------

def now_epoch_seconds_str():
    return str(int(time.time()))

def sign_request(method, path, query=None, payload=None):
    timestamp = now_epoch_seconds_str()
    query_string = "" if not query else requests.compat.urlencode(query, doseq=True)
    body = "" if not payload else requests.compat.json.dumps(payload, separators=(",", ":"))
    message = (method.upper() + timestamp + path + query_string + body).encode()
    signature = hmac.new(API_SECRET.encode(), message, hashlib.sha256).hexdigest()
    headers = {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature
    }
    return headers

def test_connection():
    path = "/v2/wallet/balances"
    url = BASE_URL + path
    headers = sign_request("GET", path)
    r = requests.get(url, headers=headers, timeout=10)
    print("Status:", r.status_code)
    print("Response:", r.json())

if __name__ == "__main__":
    test_connection()
