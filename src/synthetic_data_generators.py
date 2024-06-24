import pandas as pd
import numpy as np

def generate_providers(num_providers, num_locations, num_groups, specialties, county, min_latitude, max_latitude, min_longitude, max_longitude):

    providers = pd.DataFrame({
        "npi": [i for i in range(num_providers)],
        "specialty": np.random.choice(specialties, num_providers),
        "group_id": [np.random.choice(np.arange(num_groups)) for i in range(num_providers)],
        "efficiency": [np.random.randint(1,6) for i in range(num_providers)],
        "effectiveness": [np.random.randint(1,6) for i in range(num_providers)],
    })

    locations = pd.DataFrame({
        "location_id": [i for i in range(num_locations)],
        "county": [county] * num_locations,
        "latitude": [np.round(np.random.randint(min_latitude, max_latitude)/1000000, 6) for i in range(num_locations)],
        "longitude": [np.round(np.random.randint(min_longitude, max_longitude)/1000000, 6) for i in range(num_locations)]
    })

    df = pd.DataFrame(columns=providers.columns.tolist())

    for i in range(len(locations)):
        df.loc[i] = providers.sample(1).iloc[0]

    df = pd.concat([df, locations], axis=1)
    
    return df

def generate_members(size, min_latitude, max_latitude, min_longitude, max_longitude):
    members = pd.DataFrame({"member_id": ["".join(str(np.random.randint(10)) for i in range(10)) for j in range(size)],
                              "county": ["wayne"] * size,
                              "latitude": [np.round(np.random.randint(min_latitude,max_latitude)/1000000, 6) for i in range(size)],
                              "longitude": [np.round(np.random.randint(min_longitude, max_longitude)/1000000, 6) for i in range(size)]
                              })
    return members

synth_reqs = pd.DataFrame({"specialty": ["cardiologist", "pcp", "ent", "urologist", "obgyn"],
                          "county": ["wayne"] * 5,
                          "provider_count": [1, 2, 1, 1, 1],
                          "distance_req": [15] * 5,
                          "min_access_pct": [90] * 5,
                          "min_providers": [5, 10, 5, 5, 5]}
                          )