import requests
body = {
    "message": "Its sunny in california. The weather's just cool,,,",
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}

