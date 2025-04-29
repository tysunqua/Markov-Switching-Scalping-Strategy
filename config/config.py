import json

VN30 = {
    "auth_type": "Bearer",
    "consumerID": "0",
    "consumerSecret": "0",
    "url": "https://fc-data.ssi.com.vn/",
    "stream_url": "https://fc-datahub.ssi.com.vn/"
}

optimization_params = {}
with open("optimization/best_params.json", "r") as of:
    optimization_params = json.load(of)