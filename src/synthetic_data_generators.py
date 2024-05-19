import pandas as pd
import numpy as np

def generate_providers(size, min_latitude, max_latitude, min_longitude, max_longitude):
    providers = pd.DataFrame({"npi":["".join(str(np.random.randint(10)) for i in range(10)) for j in range(size)],
                             "specialty": np.random.choice(["cardiologist", "pcp", "ent", "urologist", "obgyn"], size),
                             "county": ["wayne"] * size,
                             "efficiency": [np.random.randint(1,6) for i in range(size)],
                             "effectiveness": [np.random.randint(1,6) for i in range(size)],
                             "latitude": [np.round(np.random.randint(min_latitude, max_latitude)/1000000, 6) for i in range(size)],
                             "longitude": [np.round(np.random.randint(min_longitude, max_longitude)/1000000, 6) for i in range(size)]}
                             )
    
    return providers

def generate_members(size, min_latitude, max_latitude, min_longitude, max_longitude):
    members = pd.DataFrame({"member_id": ["".join(str(np.random.randint(10)) for i in range(10)) for j in range(size)],
                              "county": ["wayne"] * size,
                              "latitude": [np.round(np.random.randint(min_latitude,max_latitude)/1000000, 6) for i in range(size)],
                              "longitude": [np.round(np.random.randint(min_longitude, max_longitude)/1000000, 6) for i in range(size)]
                              })
    return members

synth_reqs = pd.DataFrame({"specialty": ["cardiologist", "pcp", "ent", "urologist", "obgyn"],
                          "county": ["wayne"] * 5,
                          "distance_req": [15] * 5,
                          "min_access_pct": [90] * 5,
                          "min_providers": [5, 10, 5, 5, 5]}
                          )