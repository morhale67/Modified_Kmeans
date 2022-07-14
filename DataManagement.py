import json

import numpy as np


def load_data(data_path):
    """ Extract drivers and pickups data from json file """
    with open(data_path, 'r', encoding="utf8") as f:
        data = json.load(f)
    raw_pickups = data['latitudeAndLongitude']
    pickups = extract_relevant_info_pickups(raw_pickups)
    raw_drivers = data['cars']
    drivers, drivers_home, drivers_carload = extract_relevant_info_drivers(raw_drivers)
    return drivers, drivers_home, drivers_carload, pickups


def extract_relevant_info_drivers(raw_drivers):
    """ Extract only clustering relevant data from drivers info. Output list with id, home(lat,lon) and carload"""
    drivers = []
    for driver in raw_drivers:
        drivers.append({'id': driver['id'], 'home': [driver['home']['lat'], driver['home']['lon']], 'carload': driver['carload']})
    drivers_home = get_home(drivers)
    drivers_carload = get_carload(drivers)
    return drivers, drivers_home, drivers_carload


def extract_relevant_info_pickups(raw_pickups):
    """ Restructure pickups data to be one list of points."""
    pickups = []
    for i in range(len(raw_pickups)):
        pickup = raw_pickups[str(i)]
        pickups.append([pickup['lat'], pickup['lon']])
    return pickups


def get_home(drivers):
    drivers_home = np.array([driver['home'] for driver in drivers])
    return drivers_home


def get_carload(drivers):
    drivers_carload = np.array([driver['carload'] for driver in drivers])
    return drivers_carload


def write_clusters_to_json(cluster_labels, template_path):
    with open(template_path, 'r', encoding="utf8") as f:
        output = json.load(f)

    for i, label in enumerate(cluster_labels):
        output['Clusters']['C0']['latitudeAndLongitude'][str(i)]['cluster_labels'] = int(label)

    new_path = 'Data/clustering_output'
    with open(new_path, 'w', encoding="utf8") as f:
        json.dump(output, f)