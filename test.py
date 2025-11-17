import requests

payload = {
    "left_sev": 1,
    "right_sev": 1,
    "brain_sev": 1,
    "bounds": {
        "fio2": [21, 100],
        "peep": [5, 12],
        "inspiratory_pressure": [10, 25],
        "respiration_rate": [10, 20],
        "flow": [10, 20],
        "tidal_volume": [300, 500]
    },
    "N": 200
}

response = requests.post("http://127.0.0.1:8000/simulate", json=payload)
print(response.status_code)
print(response.json())
