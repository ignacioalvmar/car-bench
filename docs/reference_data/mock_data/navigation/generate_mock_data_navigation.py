import itertools  # Added for combinations
import json

# Removed uuid import
import math  # For finding bounds
import os  # Added for path handling
import random
import re

# from litellm import completion # Removed as not used in provided snippet
from datetime import datetime
from typing import Dict, List, Tuple  # Added Tuple

import numpy as np
from tqdm import tqdm

from car_bench.envs.car_voice_assistant.mock_data.data_manager import read_jsonl_file

# Define the center coordinates for the Gaussian distribution
CURRENT_LONGITUDE = 11.5750
CURRENT_LATITUDE = 48.1375
CURRENT_LOCATION_NAME = "Munich"
# Keep original ID format idea, ensure uniqueness via check
CURRENT_LOCATION_ID = f"loc_{CURRENT_LOCATION_NAME[:3].lower()}_9995"


CURRENT_LOCATION = {
    "id": CURRENT_LOCATION_ID,
    "name": CURRENT_LOCATION_NAME,
    "position": {"longitude": CURRENT_LONGITUDE, "latitude": CURRENT_LATITUDE},
}

# Current Time
CURRENT_TIME = datetime(
    year=2025, month=2, day=14, hour=12, minute=0
)  # can be overwritten by context init_config

# IDS to not have duplicates - use sets for O(1) lookup
LOCATION_IDS = set()
LOCATION_IDS.add(CURRENT_LOCATION_ID)
POI_IDS = set()
ROUTE_IDS = set()
# Add a separate set for plug IDs if they need cross-type uniqueness, otherwise POI_IDS might suffice
PLUG_IDS = set()


# --- Helper Functions ---


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_closest_point(long, lat, points):
    # Assuming points is a list of dicts with 'longitude', 'latitude'
    closest_point = None
    closest_distance = float("inf")  # Initialize with infinity
    closest_idx = -1
    for idx, point in enumerate(points):
        distance = haversine(
            point["latitude"], point["longitude"], lat, long
        )  # Consistent order lat, lon
        if distance < closest_distance:
            closest_distance = distance
            closest_point = point
            closest_idx = idx
    return closest_idx, closest_point, closest_distance


def generate_position_gaussian_around_point(long, lat, deviation):
    longitude = np.random.normal(long, deviation)
    latitude = np.random.normal(lat, deviation)
    return {"longitude": longitude, "latitude": latitude}


def generate_position_uniform_around_point(long, lat, radius):
    longitude = random.uniform(long - radius, long + radius)
    latitude = random.uniform(lat - radius, lat + radius)
    return {"longitude": longitude, "latitude": latitude}


def get_intermediate_point_on_great_circle(
    lon1_deg, lat1_deg, lon2_deg, lat2_deg, fraction
):
    if fraction <= 0:
        return lon1_deg, lat1_deg
    if fraction >= 1:
        return lon2_deg, lat2_deg

    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    # Calculate angular distance between points using Haversine formula components
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a_hav = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    delta = 2 * math.atan2(
        math.sqrt(a_hav), math.sqrt(1 - a_hav)
    )  # angular distance in radians

    if delta == 0:  # Points are identical
        return lon1_deg, lat1_deg

    # Interpolation formula
    A = math.sin((1 - fraction) * delta) / math.sin(delta)
    B = math.sin(fraction * delta) / math.sin(delta)

    x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
    y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
    z = A * math.sin(lat1) + B * math.sin(lat2)

    lat_i = math.atan2(z, math.sqrt(x**2 + y**2))
    lon_i = math.atan2(y, x)

    return math.degrees(lon_i), math.degrees(lat_i)


def generate_position_circular_around_point(
    center_lon_deg, center_lat_deg, max_radius_km, min_radius_km=0.0
):
    """
    Generate a random position within an annulus (min_radius_km to max_radius_km) around a center point.
    If min_radius_km=0, falls back to a disk.
    """
    if min_radius_km < 0 or min_radius_km > max_radius_km:
        raise ValueError("min_radius_km must be >= 0 and <= max_radius_km")

    # Uniformly random point in an annulus: r = sqrt(u*(R_max^2 - R_min^2) + R_min^2)
    u = random.uniform(0, 1)
    r_km = math.sqrt(u * (max_radius_km**2 - min_radius_km**2) + min_radius_km**2)
    theta_rad = random.uniform(0, 2 * math.pi)

    # Convert polar offset (r_km, theta_rad) to Cartesian offset in degrees
    KM_PER_DEG_LAT = 111.32

    delta_lat_deg = (r_km * math.cos(theta_rad)) / KM_PER_DEG_LAT

    center_lat_rad = math.radians(center_lat_deg)
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(center_lat_rad)
    if abs(km_per_deg_lon) < 1e-6:
        delta_lon_deg = 0
    else:
        delta_lon_deg = (r_km * math.sin(theta_rad)) / km_per_deg_lon

    new_lat_deg = center_lat_deg + delta_lat_deg
    new_lon_deg = center_lon_deg + delta_lon_deg

    new_lat_deg = max(-90.0, min(90.0, new_lat_deg))
    new_lon_deg = (new_lon_deg + 180) % 360 - 180

    return {"longitude": new_lon_deg, "latitude": new_lat_deg}, r_km


# --- ID Generation (Reverted to randint format) ---


def generate_unique_id_randint(prefix, num_digits, existing_ids_set):
    """Generates a unique ID with a random integer suffix."""
    max_attempts = 10**num_digits
    for _ in range(max_attempts):
        random_suffix = random.randint(10 ** (num_digits - 1), (10**num_digits) - 1)
        new_id = f"{prefix}_{random_suffix}"
        if new_id not in existing_ids_set:
            existing_ids_set.add(new_id)
            return new_id
    raise ValueError(f"Failed to generate a unique ID after {max_attempts} attempts.")


def generate_location_id(name):
    prefix = f"loc_{name[:3].lower().replace(' ', '_').replace('.', '')}"
    # Ensure CURRENT_LOCATION_ID is added if this is the first call
    if not LOCATION_IDS:
        LOCATION_IDS.add(CURRENT_LOCATION_ID)
    # Check if it's the current location trying to be added again
    # (This logic might be redundant depending on how de_cities is filtered)
    temp_id_check = f"{prefix}_{1000}"  # Just need the prefix part for comparison logic
    if temp_id_check.startswith(f"loc_{CURRENT_LOCATION_NAME[:3].lower()}"):
        # Check if the current location ID is already set and return it
        if CURRENT_LOCATION_ID in LOCATION_IDS:
            return CURRENT_LOCATION_ID  # Don't generate a new one for the base location

    # Generate unique ID for other locations
    return generate_unique_id_randint(prefix, 6, LOCATION_IDS)


def generate_poi_id(category):
    prefix = f"poi_{category.lower()[:3]}"
    return generate_unique_id_randint(prefix, 6, POI_IDS)


def generate_route_id(loc_id1, loc_id2):
    # Extract parts like 'mun', 'ber' from 'loc_mun_1234', 'loc_ber_5678'
    # rll for location to location, rlp for location to POI, rpl for POI to location
    try:
        part1 = loc_id1.split("_")[1]
        part2 = loc_id2.split("_")[1]
        rte_part1 = loc_id1.split("_")[0][0]
        rte_part2 = loc_id2.split("_")[0][0]
        prefix = f"r{rte_part1}{rte_part2}_{part1}_{part2}"
    except IndexError:
        # Fallback if ID format is unexpected
        prefix = "rte_unk_unk"
    return generate_unique_id_randint(prefix, 6, ROUTE_IDS)


def generate_plug_id():
    prefix = f"plg_cha"
    # Using PLUG_IDS set now
    return generate_unique_id_randint(prefix, 6, PLUG_IDS)


# --- POI Name Generation (Country-Aware) ---
def generate_poi_name(category, country_code="DE"):
    """Generate a POI name appropriate for the given country and category."""
    # Define name dictionaries by country code and category
    bakery_names = {
        "DE": [
            "Bäckerei Kamps",
            "Bäckerei Schäfer",
            "Der Bäcker Eifler",
            "Bäckerei Huth",
            "Bäckerei Junge",
            "Bäckerei Voigt",
            "BrotHaus",
            "Bäckerei Wimmer",
            "Hofpfisterei",
            "BackWerk",
            "Dat Backhus",
        ],
        "AT": [
            "Bäckerei Ströck",
            "Mann Brot & Gebäck",
            "Felber",
            "Anker",
            "Der Bäcker Ruetz",
            "Gradwohl",
            "Ölz Meisterbäcker",
        ],
        "FR": [
            "Boulangerie Dupain",
            "La Mie Câline",
            "Paul",
            "Brioche Dorée",
            "Au Pain Doré",
            "Maison Kayser",
            "Banette",
        ],
        "IT": [
            "Panificio Romano",
            "Il Fornaio",
            "Forno Italiano",
            "Pane e Dolci",
            "Antico Forno",
            "Panetteria Moderna",
        ],
        "ES": [
            "Panadería García",
            "Pan y Dulces",
            "El Horno",
            "La Tahona",
            "Panadería Tradicional",
            "Delicias del Pan",
        ],
        "NL": [
            "Bakkerij Bart",
            "Bakker van Maanen",
            "Le Pain Quotidien",
            "Vlaams Broodhuys",
            "Bakkerij Van Vessem",
        ],
        "BE": [
            "Bakkerij Goossens",
            "Pain Quotidien",
            "Pains et Tradition",
            "Aux Merveilleux",
            "Bloch Bakery",
        ],
        "PL": [
            "Piekarnia Cukiernia",
            "Tradycyjna Piekarnia",
            "Osiedlowa Piekarnia",
            "Chleb i Ciastka",
            "Stara Piekarnia",
        ],
        "CZ": [
            "Pekařství U Malířů",
            "Chleba Mazana",
            "Pekařství Kabát",
            "Pekařství Moravec",
            "České Pekařství",
        ],
        "HU": [
            "Lipóti Pékség",
            "Pék Cukrászda",
            "Kovács Pékség",
            "Józsa Pékség",
            "Álom Pékség",
        ],
        "SK": [
            "Pekáreň Včielka",
            "Prima Pekáreň",
            "Slopekar",
            "Pekáreň Pod Hradom",
            "Alfa Pekáreň",
        ],
        "RO": [
            "Brutăria Matei",
            "Pâine Proaspătă",
            "Brutăria Dulce",
            "Simigeria Luca",
            "Brutăria Tradițională",
        ],
        "LV": [
            "Lāču Maize",
            "Maiznīca Ķelmēni",
            "Maiznieks Rīga",
            "Bekēreja",
            "Lielezera Maiznīca",
        ],
        "CH": [
            "Bäckerei Hug",
            "Sprüngli",
            "St. Galler Brot",
            "Café Bachmann",
            "Maison Cailler",
        ],
        "default": ["Local Bakery", "Fresh Bread", "Bakery", "Bread & Pastries"],
    }

    fast_food_names = {
        "DE": [
            "Burger King",
            "McDonald's",
            "KFC",
            "Nordsee",
            "Currywurst Express",
            "Döneria",
            "Bratwurstglöckl",
        ],
        "FR": [
            "McDonald's",
            "Quick",
            "Flunch",
            "Brioche Dorée",
            "Croissanterie",
            "La Mie Câline",
            "O'Tacos",
        ],
        "IT": [
            "McDonald's",
            "Autogrill",
            "Spizzico",
            "Panzerotti",
            "Focacceria",
            "Pizza al Taglio",
            "Rosticceria",
        ],
        "ES": [
            "McDonald's",
            "Telepizza",
            "Pans & Company",
            "100 Montaditos",
            "Bocatta",
            "MásQMenos",
            "Cervecería",
        ],
        "HU": [
            "McDonald's",
            "Burger King",
            "Nordsee",
            "Wizz Burger",
            "Gyros Étterem",
            "Büfé",
            "Lángos Büfé",
        ],
        "PL": [
            "McDonald's",
            "KFC",
            "Kebab House",
            "Pizza Dominium",
            "Pijalnia Wódki i Piwa",
            "Bar Mleczny",
            "Zapiekanki",
        ],
        "CZ": [
            "McDonald's",
            "KFC",
            "Bageterie Boulevard",
            "Bramborák",
            "Kebab",
            "Rychlé Občerstvení",
            "Smažírna",
        ],
        "RO": [
            "McDonald's",
            "KFC",
            "Springtime",
            "La Cocoșatu",
            "Shaorma",
            "Gogoșerie",
            "Covrigărie",
        ],
        "NL": [
            "McDonald's",
            "Febo",
            "Smullers",
            "Cafetaria",
            "Kwekkeboom",
            "Döner Company",
            "Friet van Piet",
        ],
        "default": [
            "McDonald's",
            "Burger King",
            "KFC",
            "Subway",
            "Fast Food",
            "Quick Bite",
        ],
    }

    parking_names = {
        "DE": [
            "Parkhaus Zentrum",
            "Tiefgarage",
            "P+R Parkplatz",
            "Apcoa Parking",
            "Q-Park",
            "Parkplatz Hauptbahnhof",
        ],
        "FR": [
            "Parking Central",
            "Indigo Parking",
            "Effia Parking",
            "Q-Park",
            "Parking Souterrain",
            "Vinci Park",
        ],
        "IT": [
            "Parcheggio Centrale",
            "Apcoa Parking",
            "Parcheggio Comunale",
            "Autorimessa",
            "Parking Stazione",
        ],
        "ES": [
            "Parking Centro",
            "Saba Parking",
            "Empark",
            "Parking Municipal",
            "Aparcamiento Público",
        ],
        "HU": [
            "Parkolóház",
            "Mélygarázs",
            "Parkoló",
            "Várakozóhely",
            "Parkoló Garázs",
            "Központi Parkoló",
        ],
        "PL": [
            "Parking Miejski",
            "Strefa Parkowania",
            "Garaż Podziemny",
            "Centrum Parkingowe",
            "Parking Strzeżony",
        ],
        "CZ": [
            "Parkoviště Centrum",
            "Parkovací Dům",
            "Centrální Parkoviště",
            "Hlídané Parkoviště",
            "Garáže",
        ],
        "RO": [
            "Parcare Centrală",
            "Parcare Subterană",
            "Parcare Publică",
            "Parcare Supraetajată",
            "Parcare Beweerd",
        ],
        "CH": [
            "Parkhaus",
            "Tiefgarage",
            "Parking Souterrain",
            "Parcheggio",
            "Centre de Stationnement",
        ],
        "default": [
            "Central Parking",
            "Parking Garage",
            "Park & Ride",
            "Public Parking",
        ],
    }

    public_toilets_names = {
        "DE": [
            "Öffentliche Toilette",
            "WC-Anlage",
            "Sanitäranlagen",
            "City-WC",
            "Bahnhof Toiletten",
        ],
        "FR": [
            "Toilettes Publiques",
            "Sanisettes",
            "WC Publics",
            "Sanitaires",
            "Toilettes Municipales",
        ],
        "IT": [
            "Bagni Pubblici",
            "Servizi Igienici",
            "WC Pubblici",
            "Toilette Pubbliche",
            "Bagni Stazione",
        ],
        "ES": [
            "Aseos Públicos",
            "Servicios Públicos",
            "WC Públicos",
            "Baños Públicos",
            "Aseos Municipales",
        ],
        "HU": [
            "Nyilvános WC",
            "Mosdó",
            "Illemhely",
            "Toalett",
            "Közvécé",
            "Mellékhelyiség",
        ],
        "PL": [
            "Toaleta Publiczna",
            "WC",
            "Szalet Miejski",
            "Łazienki Publiczne",
            "Toalety",
        ],
        "CZ": [
            "Veřejné Toalety",
            "WC",
            "Veřejné Záchody",
            "Toalety",
            "Sociální Zařízení",
        ],
        "RO": [
            "Toalete Publice",
            "WC Public",
            "Grup Sanitar",
            "Toaletă",
            "Baie Publică",
        ],
        "default": ["Public Toilets", "Restrooms", "WC", "Public Lavatory"],
    }

    restaurants_names = {
        "DE": [
            "Gasthaus Zum Adler",
            "Brauhaus Germania",
            "Restaurant Ratskeller",
            "Weinstube am Markt",
            "Zum Goldenen Hahn",
        ],
        "AT": [
            "Gasthaus zur Post",
            "Wiener Wirtshaus",
            "Alpenblick",
            "Restaurant Schönbrunn",
            "Zur Goldenen Kugel",
        ],
        "FR": [
            "Bistrot de Paris",
            "Le Petit Café",
            "La Brasserie",
            "Chez Marcel",
            "Le Gourmet",
            "Café de la Place",
        ],
        "IT": [
            "Trattoria da Luigi",
            "Ristorante La Pergola",
            "Osteria del Corso",
            "Pizzeria Napoli",
            "Il Gusto Italiano",
        ],
        "ES": [
            "Restaurante El Toro",
            "La Taberna Española",
            "Mesón del Asador",
            "El Rincón de Tapas",
            "Casa Pepe",
        ],
        "NL": [
            "Eetcafé De Prins",
            "Restaurant De Tijd",
            "Brasserie Centraal",
            "Het Wapen van Amsterdam",
            "De Keuken van",
        ],
        "BE": [
            "Brasserie Grand Place",
            "Le Café des Artistes",
            "Restaurant De Haven",
            "Bij den Boer",
            "Chez Leon",
        ],
        "PL": [
            "Restauracja Staropolska",
            "Karczma Pod Kasztanami",
            "Gospoda",
            "Bar Mleczny Familia",
            "Pierogarnia",
        ],
        "CZ": [
            "Restaurace U Fleků",
            "Lokál",
            "Hospoda Na Rohu",
            "Česká Kuchyně",
            "Pivnice U Zlatého Tygra",
        ],
        "HU": [
            "Kiskakas Vendéglő",
            "Gundel Étterem",
            "Százéves Étterem",
            "Magyar Csárda",
            "Paprika Vendéglő",
        ],
        "CH": [
            "Gasthaus zum Bären",
            "Restaurant du Lac",
            "Ristorante Alpino",
            "Raclette Stube",
            "Le Chalet",
        ],
        "RO": [
            "Restaurant Hanu' lui Manuc",
            "La Ceaun",
            "Caru' cu Bere",
            "Taverna Sârbului",
            "Casa Românească",
        ],
        "LV": [
            "Restorāns Lido",
            "Latvietis",
            "Vecmeita",
            "Pie Kristapa",
            "Trīs Pavāri",
        ],
        "default": ["Restaurant", "Bistro", "Dining", "Grill House", "Café"],
    }

    supermarkets_names = {
        "DE": [
            "Edeka",
            "Rewe",
            "Lidl",
            "Aldi",
            "Penny Markt",
            "Netto Marken-Discount",
            "Kaufland",
            "Real",
        ],
        "FR": [
            "Carrefour",
            "Auchan",
            "E.Leclerc",
            "Intermarché",
            "Casino",
            "Monoprix",
            "Franprix",
            "U Express",
        ],
        "IT": [
            "Esselunga",
            "Conad",
            "Coop",
            "Carrefour",
            "Eurospin",
            "Lidl",
            "Pam",
            "Despar",
        ],
        "ES": [
            "Mercadona",
            "Carrefour",
            "Dia",
            "Lidl",
            "Eroski",
            "Alcampo",
            "El Corte Inglés",
            "Consum",
        ],
        "HU": ["CBA", "Spar", "Tesco", "Lidl", "Aldi", "Penny Market", "Príma", "Coop"],
        "PL": [
            "Biedronka",
            "Lidl",
            "Carrefour",
            "Auchan",
            "Kaufland",
            "Lewiatan",
            "Dino",
            "Stokrotka",
        ],
        "CZ": [
            "Albert",
            "Kaufland",
            "Lidl",
            "Tesco",
            "Billa",
            "Globus",
            "Penny Market",
            "Coop",
        ],
        "RO": [
            "Kaufland",
            "Carrefour",
            "Lidl",
            "Mega Image",
            "Auchan",
            "Penny",
            "Profi",
            "Cora",
        ],
        "CH": [
            "Migros",
            "Coop",
            "Denner",
            "Aldi",
            "Lidl",
            "Volg",
            "Manor Food",
            "Spar",
        ],
        "NL": ["Albert Heijn", "Jumbo", "Lidl", "Aldi", "Plus", "Dirk", "Coop", "Spar"],
        "default": ["Supermarket", "Grocery", "Food Market", "Mini Market"],
    }

    charging_stations_names = {
        "DE": [
            "EnBW",
            "Ionity",
            "Tesla Supercharger",
            "E.ON Drive",
            "Allego",
            "Fastned",
            "Ladestation",
        ],
        "FR": [
            "Ionity",
            "Tesla Supercharger",
            "Izivia",
            "Total EV Charge",
            "Engie Electric",
            "Freshmile",
            "EDF Pulse",
        ],
        "IT": [
            "Enel X",
            "Tesla Supercharger",
            "Ionity",
            "BeCharge",
            "Edison",
            "Duferco Energia",
            "Neogy",
        ],
        "ES": [
            "Iberdrola",
            "Endesa X",
            "Tesla Supercharger",
            "Ionity",
            "Repsol",
            "Wenea",
            "GIC",
        ],
        "NL": [
            "Fastned",
            "Allego",
            "NewMotion",
            "Tesla Supercharger",
            "Ionity",
            "Shell Recharge",
            "Vattenfall InCharge",
        ],
        "PL": [
            "GreenWay",
            "Orlen Charge",
            "Tesla Supercharger",
            "Ionity",
            "EV+",
            "Energa",
            "Tauron",
        ],
        "CZ": [
            "PRE",
            "ČEZ",
            "Ionity",
            "Tesla Supercharger",
            "E.ON",
            "MOL Plugee",
            "Elektromobilita",
        ],
        "HU": [
            "MOL Plugee",
            "e-Mobi",
            "Elmű",
            "Tesla Supercharger",
            "Ionity",
            "NKM Mobilitás",
            "Alte",
        ],
        "RO": [
            "Renovatio",
            "Electrica Furnizare",
            "Tesla Supercharger",
            "Ionity",
            "E.ON Drive",
            "Enel X",
            "Kaufland e-Charge",
        ],
        "CH": [
            "GOFAST",
            "Move",
            "Swisscharge",
            "Tesla Supercharger",
            "Ionity",
            "ewz",
            "Groupe E",
        ],
        "default": [
            "EV Charging Station",
            "Daily Charger",
            "Electric Vehicle Charging",
            "EV Power",
        ],
    }

    airports_names = {
        "DE": [
            "Flughafen",
            "Airport",
            "Lufthansa Terminal",
            "Air Berlin Gate",
            "Eurowings Terminal",
        ],
        "FR": [
            "Aéroport",
            "Terminal Air France",
            "Porte d'Embarquement",
            "Terminal Transavia",
            "Air France Lounge",
        ],
        "IT": [
            "Aeroporto",
            "Terminal Alitalia",
            "Porta d'Imbarco",
            "Scalo Aereo",
            "Terminal Ryanair",
        ],
        "ES": [
            "Aeropuerto",
            "Terminal Iberia",
            "Puerta de Embarque",
            "Terminal Vueling",
            "Iberia Lounge",
        ],
        "HU": ["Repülőtér", "Terminál", "Beszállókapu", "Váró", "Wizz Air Terminál"],
        "PL": ["Lotnisko", "Terminal", "Brama", "Port Lotniczy", "LOT Terminal"],
        "CZ": ["Letiště", "Terminál", "Brána", "Odletová Hala", "ČSA Terminál"],
        "RO": ["Aeroport", "Terminal", "Poartă", "Zona de Îmbarcare", "TAROM Terminal"],
        "default": ["Airport", "Terminal", "Gate", "International Airport"],
    }

    # Map linguistically similar countries (if not directly defined)
    country_map = {
        # Germanic language family
        "AT": "DE",  # Austria -> Germany
        "DK": "DE",  # Denmark -> Germany
        "CH": "DE",  # Switzerland (mainly using German names)
        # Romance language family
        "MC": "FR",  # Monaco -> France
        "AD": "ES",  # Andorra -> Spain
        "LU": "FR",  # Luxembourg -> France (also uses German)
        # Slavic language family
        "SK": "CZ",  # Slovakia -> Czech Republic
        "SI": "CZ",  # Slovenia -> Czech Republic
        "HR": "CZ",  # Croatia -> Czech Republic
        "RS": "CZ",  # Serbia -> Czech Republic
        "BY": "PL",  # Belarus -> Poland
    }

    # Resolve mapped countries
    if country_code in country_map:
        country_code = country_map[country_code]

    # Collect all category dictionaries
    category_dictionaries = {
        "bakery": bakery_names,
        "fast_food": fast_food_names,
        "parking": parking_names,
        "public_toilets": public_toilets_names,
        "restaurants": restaurants_names,
        "supermarkets": supermarkets_names,
        "charging_stations": charging_stations_names,
        "airports": airports_names,
    }

    # Get the appropriate dictionary for the category
    if category in category_dictionaries:
        # Get country-specific names or fall back to default if not available
        country_names = category_dictionaries[category].get(country_code)
        if not country_names:
            country_names = category_dictionaries[category].get("default")
            if (
                not country_names
            ):  # If default doesn't exist either, use first available country list
                for country_list in category_dictionaries[category].values():
                    if country_list:
                        country_names = country_list
                        break
                else:  # If no lists found at all
                    raise ValueError(
                        f"No name list available for category '{category}'"
                    )
        return random.choice(country_names)
    else:
        raise ValueError(f"Invalid category '{category}' for POI name generation.")


# --- Route Detail Generation ---


def generate_road_distance(lat1, lon1, lat2, lon2):
    air_distance = haversine(lat1, lon1, lat2, lon2)
    air_distance = max(0, air_distance)  # Ensure non-negative
    road_distance = air_distance * np.random.uniform(1.19, 1.21)
    return {"distance": round(road_distance, 2), "unit": "km"}


def generate_duration_hour_minutes(road_distance_km, average_speed_kph=70):
    if road_distance_km <= 0 or average_speed_kph <= 0:
        return {"hours": 0, "minutes": 0}
    time_hours = road_distance_km / average_speed_kph
    # Add some variability to the time, account that for higher time hours less percentage variability
    time_hours = time_hours * np.random.uniform(0.98, 1.02)
    hours = int(time_hours)
    minutes = math.ceil((time_hours - hours) * 60)
    return {"hours": hours, "minutes": minutes}


def generate_opening_hours(category):
    always_open = [
        "airports",
        "atm",
        "hospitals",
        "public_toilets",
        "charging_stations",
    ]
    early_open = ["bakery", "fast_food", "supermarkets"]
    mid_open = ["cafe", "restaurants"]
    flexible_open = ["parking", "petrol_stations"]

    if category in always_open:
        return "00:00h - 24:00h"

    if category in early_open:
        open_hr = random.randint(6, 8)
        close_hr = (
            random.randint(12, 16) if category == "bakery" else random.randint(18, 24)
        )
    elif category in mid_open:
        open_hr = random.randint(8, 13)
        close_hr = random.randint(17, 22)
    elif category in flexible_open:
        open_hr = random.randint(0, 8)
        close_hr = random.randint(20, 24)
    else:
        open_hr = random.randint(8, 10)
        close_hr = random.randint(18, 22)

    if close_hr <= open_hr:
        close_hr = min(open_hr + random.randint(4, 8), 24)
    return f"{open_hr:02d}:00h - {close_hr:02d}:00h"


def generate_phone_number():
    return f"+49 {random.randint(100, 999)} {random.randint(1000000, 9999999)}"


def generate_route_name_via(route_road_types):
    road_type_abbr = []
    if "highway" in route_road_types:
        road_type_abbr.append("A")
    if "urban" in route_road_types:
        road_type_abbr.extend(["L", "K"])
    if "country road" in route_road_types:
        road_type_abbr.append("B")
    if not road_type_abbr:
        road_type_abbr = ["L", "K", "B"]

    route_name_parts = set()
    num_parts = random.randint(1, 3)
    while len(route_name_parts) < num_parts:
        abbr = random.choice(road_type_abbr)
        num = random.randint(1, 999) if abbr != "A" else random.randint(1, 99)
        route_name_parts.add(f"{abbr}{num}")
    # Return as list for easier reversal later
    return sorted(list(route_name_parts))


def format_route_name_via(name_parts_list):
    """Joins the list of route name parts into a string."""
    return ", ".join(name_parts_list)


def reverse_route_name_via(name_parts_list):
    """Reverses the order of route name parts."""
    return name_parts_list[::-1]


# --- Core Generation Functions ---


# Function to generate a single route with multiple alternatives
def generate_route_alternatives(
    start_entity,
    dest_entity,
    num_alternatives=3,
    average_speed_kph=70,
    toll_road_probability=0.1,
):
    """Generates multiple route alternatives between two entities."""
    routes = []
    road_types_pool = ["highway", "urban", "country road"]

    # Basic distance/duration for reference (consistent for the primary route)
    base_distance_info = generate_road_distance(
        start_entity["position"]["latitude"],
        start_entity["position"]["longitude"],
        dest_entity["position"]["latitude"],
        dest_entity["position"]["longitude"],
    )
    base_duration_info = generate_duration_hour_minutes(
        base_distance_info["distance"], average_speed_kph
    )

    for i in range(num_alternatives):
        # Use base for first, vary slightly for others
        if i == 0:
            route_distance_info = base_distance_info
            route_duration_info = base_duration_info
        else:
            distance_multiplier = np.random.uniform(0.95, 1.05)
            speed_multiplier = np.random.uniform(0.99, 1.01)
            alt_distance = base_distance_info["distance"] * distance_multiplier
            alt_speed = average_speed_kph * speed_multiplier
            route_distance_info = {"distance": round(alt_distance, 2), "unit": "km"}
            route_duration_info = generate_duration_hour_minutes(
                alt_distance, alt_speed
            )

        route_road_types = random.sample(
            road_types_pool, random.randint(1, len(road_types_pool))
        )
        has_toll = random.random() < toll_road_probability
        if has_toll:
            route_road_types.append("includes toll road")  # Add toll explicitly

        route_name_parts = generate_route_name_via(
            route_road_types
        )  # Get list of parts

        # Use the correct start/end IDs for route ID generation
        route_id = generate_route_id(start_entity["id"], dest_entity["id"])

        route = {
            "route_id": route_id,
            "start_id": start_entity["id"],
            "destination_id": dest_entity["id"],
            "name_via_parts": route_name_parts,  # Store parts for potential reversal
            "name_via": format_route_name_via(route_name_parts),  # Formatted string
            "distance_km": route_distance_info["distance"],
            "duration_hours": route_duration_info["hours"],
            "duration_minutes": route_duration_info["minutes"],
            "road_types": sorted(list(set(route_road_types))),
            "includes_toll": has_toll,
            "alias": [],
        }
        routes.append(route)

    # Sort routes by duration (fastest first)
    routes.sort(key=lambda r: (r["duration_hours"], r["duration_minutes"]))

    # Add aliases
    if routes:
        routes[0]["alias"].append("fastest")
        routes[0]["alias"].append("first")
        if len(routes) > 1:
            routes[1]["alias"].append("second")
        if len(routes) > 2:
            routes[2]["alias"].append("third")

        shortest_route_idx = min(
            range(len(routes)), key=lambda i: routes[i]["distance_km"]
        )
        if "shortest" not in routes[shortest_route_idx]["alias"]:
            routes[shortest_route_idx]["alias"].append("shortest")

    return routes


def create_symmetric_routes(original_routes, new_start_id, new_dest_id):
    """Creates symmetric routes based on original routes."""
    symmetric_routes = []
    for original_route in original_routes:
        # Deep copy to avoid modifying original
        symmetric_route = json.loads(json.dumps(original_route))

        # Swap source and destination
        symmetric_route["start_id"] = new_start_id
        symmetric_route["destination_id"] = new_dest_id

        # Generate new ID for the reversed direction
        symmetric_route["route_id"] = generate_route_id(new_start_id, new_dest_id)

        # Reverse the name_via parts and format
        reversed_name_parts = reverse_route_name_via(original_route["name_via_parts"])
        symmetric_route["name_via_parts"] = reversed_name_parts
        symmetric_route["name_via"] = format_route_name_via(reversed_name_parts)

        # Keep distance, duration, road_types, toll status same
        # Keep aliases (fastest/shortest relative to this set of symmetric alternatives)

        symmetric_routes.append(symmetric_route)

    # Re-sort and re-apply aliases relative to the new direction's alternatives
    symmetric_routes.sort(key=lambda r: (r["duration_hours"], r["duration_minutes"]))
    if symmetric_routes:
        # Clear existing aliases first
        for r in symmetric_routes:
            r["alias"] = []

        symmetric_routes[0]["alias"].append("fastest")
        symmetric_routes[0]["alias"].append("first")
        if len(symmetric_routes) > 1:
            symmetric_routes[1]["alias"].append("second")
        if len(symmetric_routes) > 2:
            symmetric_routes[2]["alias"].append("third")

        shortest_route_idx = min(
            range(len(symmetric_routes)),
            key=lambda i: symmetric_routes[i]["distance_km"],
        )
        if "shortest" not in symmetric_routes[shortest_route_idx]["alias"]:
            symmetric_routes[shortest_route_idx]["alias"].append("shortest")

    return symmetric_routes


def create_route_metadata(
    start_id, dest_id, base_route_id, fraction=None, detour_distance_km=None
):
    """Creates metadata for dynamically generating routes."""
    route_id = generate_route_id(start_id, dest_id)

    metadata = {
        "route_id": route_id,
        "start_id": start_id,
        "destination_id": dest_id,
        "base_route_id": base_route_id,
    }

    if fraction is not None:
        metadata["fraction"] = round(fraction, 3)

    if detour_distance_km is not None:
        metadata["detour_distance_km"] = detour_distance_km

    return metadata, route_id


# Function to generate POIs for a given location (RETURNS ONLY POIs)
def generate_poi_for_location(
    location, airport_details, locations_dict, num_pois_per_category=2
):
    """Generates POIs for a single location. Returns a list of POI dicts."""
    categories = [
        "airports",
        "bakery",
        "fast_food",
        "parking",
        "public_toilets",
        "restaurants",
        "supermarkets",
        "charging_stations",
    ]
    generated_pois = []  # POIs generated specifically for this location

    # Get location ISO code (default to 'DE' if not found)
    loc_country_code = "DE"  # Default
    if location["id"] in locations_dict:
        loc_country_code = locations_dict[location["id"]].get("iso2", "DE")

    # --- Handle Airports ---
    closest_airport_idx, closest_airport_point, closest_dist = find_closest_point(
        location["position"]["longitude"],
        location["position"]["latitude"],
        [d["position"] for d in airport_details],  # Pass only positions
    )

    if closest_airport_idx != -1:
        closest_airport_detail = airport_details[closest_airport_idx]
        airport_poi = {
            "id": closest_airport_detail["id"],
            "name": closest_airport_detail["name"],
            "category": "airports",
            "position": closest_airport_detail["position"],
            "opening_hours": "00:00h - 24:00h",
            "phone_number": generate_phone_number(),
            "corresponding_location_id": location[
                "id"
            ],  # Link POI to its 'closest' location
        }
        generated_pois.append(airport_poi)
        # Add ID to global set here if managing unique POIs globally
        POI_IDS.add(airport_poi["id"])

    # --- Handle Other Categories ---
    category_radius = {
        "atm": 0.03,
        "bakery": 0.03,
        "cafe": 0.03,
        "fast_food": 0.05,
        "parking": 0.03,
        "public_toilets": 0.05,
        "restaurants": 0.03,
        "supermarkets": 0.03,
        "charging_stations": 0.06,
    }

    for category in categories:
        if category == "airports":
            continue

        radius = category_radius.get(category, 0.04)

        for _ in range(num_pois_per_category):
            poi_id = generate_poi_id(
                category
            )  # Generates unique ID and adds to POI_IDS
            poi_position = generate_position_uniform_around_point(
                location["position"]["longitude"],
                location["position"]["latitude"],
                radius,
            )
            poi_name = generate_poi_name(category, loc_country_code)
            poi_opening_hours = generate_opening_hours(category)
            poi_phone = generate_phone_number()

            poi_data = {
                "id": poi_id,
                "name": poi_name,
                "category": category,
                "position": poi_position,
                "opening_hours": poi_opening_hours,
                "phone_number": poi_phone,
                "corresponding_location_id": location[
                    "id"
                ],  # Link to generating location
            }

            if category == "charging_stations":
                plugs = []
                num_plugs = random.randint(1, 5)
                for _ in range(num_plugs):
                    plug_id = (
                        generate_plug_id()
                    )  # Generates unique ID and adds to PLUG_IDS
                    power_type = random.choice(["AC", "DC"])
                    power_kw = (
                        random.choice([11, 22])
                        if power_type == "AC"
                        else random.choice([50, 100, 150, 200, 250, 300, 350])
                    )
                    availability = random.choices(
                        ["available", "occupied", "maintenance"],
                        weights=[0.6, 0.35, 0.05],
                        k=1,
                    )[0]
                    plugs.append(
                        {
                            "plug_id": plug_id,
                            "power_type": power_type,
                            "power_kw": power_kw,
                            "availability": availability,
                        }
                    )
                poi_data["charging_plugs"] = plugs

            generated_pois.append(poi_data)

    return generated_pois


# Function to generate weather data (Unchanged from previous refactor)
def generate_weather():
    times = [
        "00:00",
        "03:00",
        "06:00",
        "09:00",
        "12:00",
        "15:00",
        "18:00",
        "21:00",
        "24:00",
    ]
    conditions = [
        "sunny",
        "cloudy",
        "partly_cloudy",
        "cloudy_and_rain",
        "cloudy_and_rain_and_hail",
        "cloudy_and_rain_and_thunderstorm",
        "cloudy_and_fog",
    ]
    weather_data = []
    temp_base = random.randint(-10, 30)
    wind_base = max(0, int(np.random.normal(10, 8)))
    humidity_base = random.randint(40, 95)
    condition_base = random.choice(conditions)
    weather_noon = {
        "temperature_c": temp_base,
        "wind_speed_kph": wind_base,
        "humidity_percent": humidity_base,
        "condition": condition_base,
    }

    current_weather = weather_noon.copy()
    for i in range(times.index("12:00"), len(times) - 1):
        weather_data.append(
            {"start_time": times[i], "end_time": times[i + 1], **current_weather}
        )
        temp_change = (
            random.randint(-3, 0) if i > times.index("15:00") else random.randint(-2, 1)
        )
        wind_change = random.randint(-4, 2)
        humidity_change = (
            random.randint(-2, 3)
            if current_weather["condition"]
            not in [
                "cloudy_and_rain",
                "cloudy_and_rain_and_hail",
                "cloudy_and_rain_and_thunderstorm",
                "cloudy_and_fog",
            ]
            else random.randint(0, 5)
        )
        current_weather["temperature_c"] += temp_change
        current_weather["wind_speed_kph"] = max(
            0, current_weather["wind_speed_kph"] + wind_change
        )
        current_weather["humidity_percent"] = max(
            30, min(100, current_weather["humidity_percent"] + humidity_change)
        )
        if random.random() < 0.15:
            current_weather["condition"] = random.choice(conditions)

    current_weather = weather_noon.copy()
    for i in range(times.index("12:00") - 1, -1, -1):
        temp_change = (
            random.randint(-1, 2) if i < times.index("06:00") else random.randint(0, 3)
        )
        wind_change = random.randint(-3, 3)
        humidity_change = (
            random.randint(-4, 1)
            if current_weather["condition"]
            not in [
                "cloudy_and_rain",
                "cloudy_and_rain_and_hail",
                "cloudy_and_rain_and_thunderstorm",
                "cloudy_and_fog",
            ]
            else random.randint(-5, 0)
        )
        current_weather["temperature_c"] -= temp_change
        current_weather["wind_speed_kph"] = max(
            0, current_weather["wind_speed_kph"] - wind_change
        )
        current_weather["humidity_percent"] = max(
            30, min(100, current_weather["humidity_percent"] - humidity_change)
        )
        if random.random() < 0.15:
            current_weather["condition"] = random.choice(conditions)
        weather_data.insert(
            0, {"start_time": times[i], "end_time": times[i + 1], **current_weather}
        )

    if len(weather_data) != 8:
        print(
            f"Warning: Generated {len(weather_data)} weather intervals instead of 8. Adjusting."
        )
        while len(weather_data) < 8:
            weather_data.append(
                weather_data[-1] if weather_data else weather_noon
            )  # Basic fill
        weather_data = weather_data[:8]
    return weather_data


# --- Main Execution ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Output goes into the same directory as the script
    output_dir = script_dir
    nav_output_dir = output_dir  # Output specifically to this directory
    os.makedirs(nav_output_dir, exist_ok=True)

    # --- Define Output Files ---
    locations_file = os.path.join(nav_output_dir, "locations.jsonl")
    pois_file = os.path.join(nav_output_dir, "pois.jsonl")
    weather_file = os.path.join(nav_output_dir, "weather.jsonl")
    routes_loc_loc_file = os.path.join(nav_output_dir, "routes_location_location.jsonl")
    routes_loc_poi_file = os.path.join(nav_output_dir, "routes_location_poi.jsonl")
    routes_poi_loc_file = os.path.join(nav_output_dir, "routes_poi_location.jsonl")
    routes_metadata_file = os.path.join(nav_output_dir, "routes_metadata.jsonl")

    output_files = [
        locations_file,
        pois_file,
        weather_file,
        routes_loc_loc_file,
        routes_loc_poi_file,
        routes_poi_loc_file,
        routes_metadata_file,
    ]

    # --- Clean Existing Files ---
    print("Cleaning existing output files...")
    for f_path in output_files:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"Warning: Could not remove file {f_path}. Error: {e}")

    # --- Load Input Data (Locations, Airports) ---
    print("Loading input data...")
    locations = []
    locations.append(CURRENT_LOCATION)  # Add home location first
    LOCATION_IDS.add(CURRENT_LOCATION_ID)  # Ensure home ID is in the set

    # Load German cities
    try:
        # cities_file_path = os.path.join(script_dir, "de_cities.jsonc")
        cities_file_path = os.path.join(script_dir, "europe_cities.jsonc")
        with open(cities_file_path, "r", encoding="utf-8") as f:
            jsonc_text = f.read()
            json_text = re.sub(r"//.*?\n", "\n", jsonc_text)
            json_text = re.sub(r"/\*.*?\*/", "", json_text, flags=re.DOTALL)
            de_cities = json.loads(json_text)

        print(f"Loaded {len(de_cities)} cities.")
        num_cities_to_add = 1000  # Target 129 total
        # num_cities_to_add = 4 # DEBUG: Use fewer cities for faster testing
        available_cities = [
            city for city in de_cities if city.get("city") != CURRENT_LOCATION_NAME
        ]
        num_cities_to_add = min(num_cities_to_add, len(available_cities))
        # cities_to_process = random.sample(available_cities, num_cities_to_add)
        cities_to_process = available_cities[
            :num_cities_to_add
        ]  # Take first N for consistency
        locations_dict = {}
        for city_data in cities_to_process:
            city_name = city_data.get("city")
            lng = city_data.get("lng")
            lat = city_data.get("lat")
            iso2 = city_data.get("iso2", "DE")

            if city_name and lng is not None and lat is not None:
                # generate_location_id adds the ID to LOCATION_IDS set
                location_id = generate_location_id(city_name)
                location = {
                    "id": location_id,
                    "name": city_name,
                    "position": {"longitude": float(lng), "latitude": float(lat)},
                    "iso2": iso2,
                }
                locations.append(location)
                locations_dict[location_id] = location
            else:
                print(f"Warning: Skipping city data: {city_data}")

        locations_dict[CURRENT_LOCATION_ID] = {
            "id": CURRENT_LOCATION_ID,
            "name": CURRENT_LOCATION_NAME,
            "position": {"longitude": CURRENT_LONGITUDE, "latitude": CURRENT_LATITUDE},
            "iso2": "DE",  # Assuming Munich is in Germany
        }

    except Exception as e:
        print(f"Error loading cities: {e}")

    print(f"Total locations to process: {len(locations)}")

    # Load German airports
    airport_details = []
    try:
        airports_file_path = os.path.join(script_dir, "de_airports.json")
        with open(airports_file_path, "r", encoding="utf-8") as f:
            de_airports = json.load(f)
        print(f"Loaded {len(de_airports)} airports.")
        for airport_data in de_airports:
            name = airport_data.get("name")
            lon = airport_data.get("longitude")
            lat = airport_data.get("latitude")
            if name and lon is not None and lat is not None:
                # generate_poi_id adds airport POI ID to POI_IDS set
                airport_id = generate_poi_id("airports")
                airport_details.append(
                    {
                        "id": airport_id,
                        "name": name,
                        "position": {"longitude": float(lon), "latitude": float(lat)},
                    }
                )
            else:
                print(f"Warning: Skipping airport data: {airport_data}")
    except Exception as e:
        print(f"Error loading airports: {e}")

    print(f"Using {len(airport_details)} airports for POI generation.")

    # --- Generation Phase 1: Locations, Dependent POIs, Weather ---
    print("\n--- Phase 1: Generating Locations, Location-Dependent POIs, Weather ---")
    all_generated_pois = (
        {}
    )  # Dict: {poi_id: poi_data} - Will include ALL POIs eventually

    # Write locations, location-dependent POIs, weather
    with open(locations_file, "w", encoding="utf-8") as f_loc, open(
        pois_file, "w", encoding="utf-8"
    ) as f_pois, open(weather_file, "w", encoding="utf-8") as f_weather:

        poi_write_count = 0
        for location in tqdm(locations, desc="Phase 1: Locations, Dep. POIs, Weather"):
            # 1. Write Location
            f_loc.write(json.dumps(location) + "\n")
            # 2. Generate POIs dependent on this Location
            location_pois = generate_poi_for_location(
                location, airport_details, locations_dict
            )
            # 3. Store/Write Unique POIs generated in this step
            for poi in location_pois:
                if poi["id"] not in all_generated_pois:
                    all_generated_pois[poi["id"]] = poi
                    poi_to_write = poi.copy()
                    poi_to_write.pop("name_via_parts", None)
                    f_pois.write(json.dumps(poi_to_write) + "\n")
                    poi_write_count += 1
            # 4. Generate & Write Weather
            weather_data = generate_weather()
            location_weather = {
                "location_id": location["id"],
                "location_name": location["name"],
                "weather": weather_data,
            }
            f_weather.write(json.dumps(location_weather) + "\n")
        print(
            f"Phase 1: Wrote {poi_write_count} location-dependent/airport POIs to {os.path.basename(pois_file)}"
        )

    # # --- *** NEW: Phase 1b: Location-Independent POIs *** ---
    # print("\n--- Phase 1b: Generating Location-Independent POIs ---")
    # independent_poi_count = 0
    # if not locations:
    #     print("Warning: No locations loaded, cannot generate independent POIs.")
    # else:
    #     # 1. Determine geographic bounds from loaded locations
    #     min_lon = min(loc['position']['longitude'] for loc in locations)
    #     max_lon = max(loc['position']['longitude'] for loc in locations)
    #     min_lat = min(loc['position']['latitude'] for loc in locations)
    #     max_lat = max(loc['position']['latitude'] for loc in locations)
    #     print(f"Geographic bounds for independent POIs: Lon ({min_lon:.4f}, {max_lon:.4f}), Lat ({min_lat:.4f}, {max_lat:.4f})")

    #     # Categories for independent POIs (exclude airports)
    #     independent_categories = ["atm", "bakery", "cafe", "fast_food", "parking", "public_toilets", "restaurants", "supermarkets", "charging_stations"]
    #     num_locations = len(locations)

    #     # 2. Open POI file in append mode
    #     with open(pois_file, "a", encoding='utf-8') as f_pois_append:
    #         for category in tqdm(independent_categories, desc="Phase 1b: Categories"):
    #             pois_added_this_category = 0
    #             # 4. Generate num_locations independent POIs per category
    #             for _ in range(num_locations):
    #                 # 5. Generate uniform position within bounds
    #                 poi_lon = random.uniform(min_lon, max_lon)
    #                 poi_lat = random.uniform(min_lat, max_lat)
    #                 poi_position = {"longitude": poi_lon, "latitude": poi_lat}

    #                 # 6. Generate POI details
    #                 try:
    #                     poi_id = generate_poi_id(category) # Ensures uniqueness via POI_IDS set
    #                     poi_name = generate_poi_name(category)
    #                     poi_opening_hours = generate_opening_hours(category)
    #                     poi_phone = generate_phone_number()
    #                 except ValueError as e: # Catch error from generate_poi_name if category is invalid
    #                      print(f"Warning: Skipping independent POI generation for potentially invalid category '{category}': {e}")
    #                      break # Skip rest of generation for this category

    #                 # 7. Find closest location
    #                 # Need location positions for find_closest_point
    #                 location_positions = [{"longitude": loc['position']['longitude'], "latitude": loc['position']['latitude']} for loc in locations]
    #                 closest_idx, _, _ = find_closest_point(poi_lon, poi_lat, location_positions)
    #                 if closest_idx != -1:
    #                      closest_location_id = locations[closest_idx]['id']
    #                 else:
    #                      # Fallback if no locations found (shouldn't happen if locations list is not empty)
    #                      print("Warning: Could not find closest location for an independent POI. Skipping assignment.")
    #                      closest_location_id = None # Or assign a default?

    #                 # 8. Create POI dict
    #                 poi_data = {
    #                     "id": poi_id,
    #                     "name": poi_name,
    #                     "category": category,
    #                     "position": poi_position,
    #                     "opening_hours": poi_opening_hours,
    #                     "phone_number": poi_phone,
    #                     "corresponding_location_id": closest_location_id, # Assign closest
    #                 }

    #                 # Handle charging station specifics if category matches
    #                 if category == "charging_stations":
    #                     plugs = []
    #                     num_plugs = random.randint(1, 5)
    #                     for _ in range(num_plugs):
    #                         plug_id = generate_plug_id()
    #                         power_type = random.choice(['AC', 'DC'])
    #                         power_kw = random.choice([11, 22]) if power_type == 'AC' else random.choice([50, 100, 150, 200, 250, 300, 350])
    #                         availability = random.choice(['available', 'occupied', 'maintenance'])
    #                         plugs.append({"plug_id": plug_id, "power_type": power_type, "power_kw": power_kw, "availability": availability})
    #                     poi_data["charging_plugs"] = plugs

    #                 # 9. Add to master dictionary (ensures routes get generated for it later)
    #                 # Check uniqueness again before adding, although generate_poi_id should handle it
    #                 if poi_id not in all_generated_pois:
    #                     all_generated_pois[poi_id] = poi_data
    #                     # 10. Write to POI file (append mode)
    #                     f_pois_append.write(json.dumps(poi_data) + "\n")
    #                     independent_poi_count += 1
    #                     pois_added_this_category += 1
    #                 else:
    #                      print(f"Warning: Duplicate POI ID detected and skipped during independent generation: {poi_id}")

    #             # print(f"Added {pois_added_this_category} independent POIs for category '{category}'") # Optional detail

    #     print(f"Phase 1b: Added {independent_poi_count} location-independent POIs.")
    #     print(f"Total unique POIs after Phase 1b: {len(all_generated_pois)}")

    # --- Generation Phase 2: Location -> Location Routes (Symmetric) ---
    print("\n--- Phase 2: Generating Location -> Location Routes ---")
    loc_loc_route_count = 0
    # Store ALL route alternatives, not just the fastest one
    all_loc_loc_routes_details = (
        {}
    )  # Format: {(loc1_id, loc2_id): [route1, route2, route3]}
    with open(routes_loc_loc_file, "w", encoding="utf-8") as f_routes_ll:
        location_pairs = itertools.combinations(locations, 2)
        num_pairs = len(locations) * (len(locations) - 1) // 2

        for loc1, loc2 in tqdm(
            location_pairs, desc="Phase 2: Loc-Loc Routes", total=num_pairs
        ):
            # Generate A -> B routes
            routes_ab = generate_route_alternatives(
                loc1, loc2, average_speed_kph=80, num_alternatives=3
            )

            # Store ALL routes, not just the fastest one
            if routes_ab:
                all_loc_loc_routes_details[(loc1["id"], loc2["id"])] = []
                for route in routes_ab:
                    # Add essential details for POI generation
                    route_details = {
                        "start_pos": loc1["position"],
                        "end_pos": loc2["position"],
                        "distance_km": route["distance_km"],
                        "start_id": loc1["id"],
                        "end_id": loc2["id"],
                        "road_types": route["road_types"],
                        "name_via_parts": route.get("name_via_parts", ["B1"]),
                        "route_id": route["route_id"],
                    }
                    all_loc_loc_routes_details[(loc1["id"], loc2["id"])].append(
                        route_details
                    )

                    # Write to file
                    route_to_write = route.copy()
                    route_to_write.pop("name_via_parts", None)  # Clean helper field
                    f_routes_ll.write(json.dumps(route_to_write) + "\n")
                    loc_loc_route_count += 1

            # Generate symmetric B -> A routes based on A -> B
            # create_symmetric_routes requires new start/end IDs
            routes_ba = create_symmetric_routes(routes_ab, loc2["id"], loc1["id"])

            # Store BA routes as well
            if routes_ba:
                all_loc_loc_routes_details[(loc2["id"], loc1["id"])] = []
                for route in routes_ba:
                    route_details = {
                        "start_pos": loc2["position"],
                        "end_pos": loc1["position"],
                        "distance_km": route["distance_km"],
                        "start_id": loc2["id"],
                        "end_id": loc1["id"],
                        "road_types": route["road_types"],
                        "name_via_parts": route.get("name_via_parts", ["B1"]),
                        "route_id": route["route_id"],
                    }
                    all_loc_loc_routes_details[(loc2["id"], loc1["id"])].append(
                        route_details
                    )

                    route_to_write = route.copy()
                    route_to_write.pop("name_via_parts", None)  # Clean helper field
                    f_routes_ll.write(json.dumps(route_to_write) + "\n")
                    loc_loc_route_count += 1

        print(
            f"Phase 2: Wrote {loc_loc_route_count} route alternatives to {os.path.basename(routes_loc_loc_file)}"
        )
        print(
            f"Phase 2: Stored {len(all_loc_loc_routes_details)} location pairs with {sum(len(routes) for routes in all_loc_loc_routes_details.values())} total routes for POI generation"
        )

    # --- Phase 2b: Generating POIs Along FASTEST Loc-Loc Routes Only ---
    print(
        "\n--- Phase 2b: Generating POIs Along Fastest Loc-Loc Routes & Their Routes ---"
    )
    pois_along_routes_count = 0
    poi_ids_along_routes = set()  # To track generated POIs along routes
    routes_metadata_count = 0
    already_processed_routes = set()  # To track generated routes and avoid duplicates
    processed_location_pairs = (
        set()
    )  # To track processed location pairs (loc1, loc2) to avoid reverse duplicates
    independent_categories = [
        "bakery",
        "fast_food",
        "parking",
        "public_toilets",
        "restaurants",
        "supermarkets",
        "charging_stations",
    ]

    # Add routes_metadata_file to outputs
    routes_metadata_file = os.path.join(nav_output_dir, "routes_metadata.jsonl")
    output_files.append(routes_metadata_file)

    if not locations or not all_loc_loc_routes_details:
        print(
            "Warning: No locations or routes available, cannot generate POIs along routes."
        )
    else:
        # Create empty files for compatibility
        with open(routes_loc_poi_file, "w", encoding="utf-8") as f_loc_poi, open(
            routes_poi_loc_file, "w", encoding="utf-8"
        ) as f_poi_loc:
            pass

        with open(pois_file, "a", encoding="utf-8") as f_pois_append, open(
            routes_metadata_file, "w", encoding="utf-8"
        ) as f_metadata:

            for (loc1_id, loc2_id), route_alternatives in tqdm(
                all_loc_loc_routes_details.items(),
                desc="Phase 2b: POIs Along Routes",
                total=len(all_loc_loc_routes_details),
            ):

                # Skip if we've already processed this location pair in reverse order
                location_pair = tuple(sorted([loc1_id, loc2_id]))
                if location_pair in processed_location_pairs:
                    continue

                processed_location_pairs.add(location_pair)

                # Only use the fastest route (first one) to generate POIs
                if not route_alternatives:
                    continue

                # The fastest route is the first one in the list (assuming routes are sorted by duration)
                fastest_route = route_alternatives[0]

                loc1_pos = fastest_route["start_pos"]
                loc2_pos = fastest_route["end_pos"]
                route_distance_km = fastest_route["distance_km"]
                base_road_types = fastest_route["road_types"]
                base_name_via_parts = fastest_route["name_via_parts"]

                if route_distance_km <= 0:
                    continue

                # Calculate how many POIs to place along this route
                num_pois_on_route = math.floor(route_distance_km / 50.0)

                if num_pois_on_route > 0:
                    for i in range(1, num_pois_on_route + 1):
                        target_distance_on_great_circle_km = i * 50.0

                        fraction = (
                            target_distance_on_great_circle_km / route_distance_km
                        )
                        fraction = round(
                            fraction, 3
                        )  # Round to 3 decimal places for consistency
                        if fraction >= 1.0:
                            fraction = 0.999

                        # Fix the typo in the parameter - correct the second 'longitude' to 'latitude'
                        poi_center_lon, poi_center_lat = (
                            get_intermediate_point_on_great_circle(
                                loc1_pos["longitude"],
                                loc1_pos["latitude"],
                                loc2_pos["longitude"],
                                loc2_pos["latitude"],
                                fraction,
                            )
                        )

                        # For each category, create a POI and its routes
                        for category in independent_categories:

                            # Create slightly different positions for each category
                            category_poi_position, r_km = (
                                generate_position_circular_around_point(
                                    poi_center_lon,
                                    poi_center_lat,
                                    max_radius_km=5.0,
                                    min_radius_km=1.0,
                                )
                            )

                            # Determine which endpoint is closer for corresponding_location_id
                            dist_to_loc1_endpoint = haversine(
                                category_poi_position["latitude"],
                                category_poi_position["longitude"],
                                loc1_pos["latitude"],
                                loc1_pos["longitude"],
                            )
                            dist_to_loc2_endpoint = haversine(
                                category_poi_position["latitude"],
                                category_poi_position["longitude"],
                                loc2_pos["latitude"],
                                loc2_pos["longitude"],
                            )
                            # Get country codes for the endpoints
                            loc1_country_code = locations_dict.get(loc1_id, {}).get(
                                "iso2", "DE"
                            )
                            loc2_country_code = locations_dict.get(loc2_id, {}).get(
                                "iso2", "DE"
                            )

                            # Use the country code of the closest location
                            poi_country_code = (
                                loc1_country_code
                                if dist_to_loc1_endpoint <= dist_to_loc2_endpoint
                                else loc2_country_code
                            )

                            try:
                                poi_id = generate_poi_id(category)
                                poi_name = generate_poi_name(category, poi_country_code)
                                poi_opening_hours = generate_opening_hours(category)
                                poi_phone = generate_phone_number()
                                poi_ids_along_routes.add(
                                    poi_id
                                )  # Track POI ID for this route
                            except ValueError as e:
                                print(
                                    f"Warning: Skipping POI along route for category '{category}': {e}"
                                )
                                continue

                            forward_route_ids = [
                                route["route_id"] for route in route_alternatives
                            ]
                            reverse_routes = all_loc_loc_routes_details.get(
                                (loc2_id, loc1_id), []
                            )
                            reverse_route_ids = [
                                route["route_id"] for route in reverse_routes
                            ]
                            corresponding_route_ids = (
                                forward_route_ids + reverse_route_ids
                            )

                            poi_data = {
                                "id": poi_id,
                                "name": poi_name,
                                "category": category,
                                "position": category_poi_position,
                                "opening_hours": poi_opening_hours,
                                "phone_number": poi_phone,
                                "corresponding_route_ids": corresponding_route_ids,
                                "route_positions": {
                                    forward_route_ids[0][:11]: {
                                        "at_route_kilometer": round(
                                            target_distance_on_great_circle_km, 1
                                        )
                                    },
                                    (
                                        reverse_route_ids[0][:11]
                                        if reverse_route_ids
                                        else "none"
                                    ): {
                                        "at_route_kilometer": round(
                                            route_distance_km
                                            - target_distance_on_great_circle_km,
                                            1,
                                        )
                                    },
                                },
                            }

                            if category == "charging_stations":
                                plugs = []
                                num_plugs = random.randint(1, 5)
                                for _plug_idx in range(num_plugs):
                                    plug_id_val = generate_plug_id()
                                    power_type = random.choice(["AC", "DC"])
                                    power_kw = (
                                        random.choice([11, 22])
                                        if power_type == "AC"
                                        else random.choice(
                                            [50, 100, 150, 200, 250, 300, 350]
                                        )
                                    )
                                    availability = random.choices(
                                        ["available", "occupied", "maintenance"],
                                        weights=[0.6, 0.35, 0.05],
                                        k=1,
                                    )[0]
                                    plugs.append(
                                        {
                                            "plug_id": plug_id_val,
                                            "power_type": power_type,
                                            "power_kw": power_kw,
                                            "availability": availability,
                                        }
                                    )
                                poi_data["charging_plugs"] = plugs

                            # Add POI to all_generated_pois and write to file
                            if poi_id not in all_generated_pois:
                                all_generated_pois[poi_id] = poi_data
                                pois_along_routes_count += 1

                                # Instead of generating full routes, create metadata
                                # Get location objects for route generation (needed for detour calculation)
                                loc1_entity = next(
                                    (loc for loc in locations if loc["id"] == loc1_id),
                                    None,
                                )
                                loc2_entity = next(
                                    (loc for loc in locations if loc["id"] == loc2_id),
                                    None,
                                )

                                if loc1_entity and loc2_entity:
                                    # Calculate detour values based on fraction of main route
                                    detour_distance_km = r_km
                                    detour_distance_km = round(
                                        detour_distance_km, 2
                                    )  # Round to 2 decimal places

                                    # For loc1 to POI routes
                                    if (
                                        loc1_id,
                                        poi_id,
                                    ) not in already_processed_routes:
                                        # Process all route alternatives for loc1 to POI
                                        for route_idx, route_alt in enumerate(
                                            route_alternatives
                                        ):
                                            base_route_id = route_alt[
                                                "route_id"
                                            ]  # loc1→loc2 route
                                            # Generate route ID
                                            route_id = generate_route_id(
                                                loc1_id, poi_id
                                            )
                                            # Create metadata for this route
                                            metadata = {
                                                "route_id": route_id,
                                                "start_id": loc1_id,
                                                "destination_id": poi_id,
                                                "base_route_id": base_route_id,
                                                "fraction": fraction,
                                                "detour_distance_km": detour_distance_km,
                                                "route_alternative": route_idx,
                                                "on_route": True,
                                                "is_reverse": False,
                                            }

                                            # Write metadata to file
                                            f_metadata.write(
                                                json.dumps(metadata) + "\n"
                                            )
                                            routes_metadata_count += 1

                                            # Get corresponding reverse route (loc2→loc1)
                                            reverse_routes = (
                                                all_loc_loc_routes_details.get(
                                                    (loc2_id, loc1_id), []
                                                )
                                            )
                                            reverse_base_id = (
                                                reverse_routes[route_idx]["route_id"]
                                                if reverse_routes
                                                and route_idx < len(reverse_routes)
                                                else None
                                            )

                                            if reverse_base_id:
                                                # Create symmetric POI to loc1 route metadata using loc2→loc1 base route
                                                poi_to_loc1_route_id = (
                                                    generate_route_id(poi_id, loc1_id)
                                                )
                                                poi_to_loc1_metadata = {
                                                    "route_id": poi_to_loc1_route_id,
                                                    "start_id": poi_id,
                                                    "destination_id": loc1_id,
                                                    "base_route_id": reverse_base_id,  # Use loc2→loc1 route
                                                    "fraction": fraction,
                                                    "detour_distance_km": detour_distance_km,
                                                    "route_alternative": route_idx,
                                                    "on_route": True,
                                                    "is_reverse": False,
                                                }

                                                f_metadata.write(
                                                    json.dumps(poi_to_loc1_metadata)
                                                    + "\n"
                                                )
                                                routes_metadata_count += 1

                                        # Mark as processed after all alternatives
                                        already_processed_routes.add((loc1_id, poi_id))
                                        already_processed_routes.add((poi_id, loc1_id))

                                    # For loc2 to POI routes - also use all route alternatives
                                    if (
                                        loc2_id,
                                        poi_id,
                                    ) not in already_processed_routes:
                                        # Get the route alternatives for loc2 to loc1
                                        reverse_routes = all_loc_loc_routes_details.get(
                                            (loc2_id, loc1_id), []
                                        )

                                        # Process all route alternatives for loc2 to POI
                                        for route_idx, route_alt in enumerate(
                                            reverse_routes
                                            if reverse_routes
                                            else route_alternatives
                                        ):
                                            # Use appropriate base route ID
                                            if reverse_routes:
                                                base_route_id = route_alt["route_id"]
                                                is_true_reverse = False
                                            else:
                                                base_route_id = route_alternatives[
                                                    min(
                                                        route_idx,
                                                        len(route_alternatives) - 1,
                                                    )
                                                ]["route_id"]
                                                is_true_reverse = True

                                            # Generate route ID
                                            route_id = generate_route_id(
                                                loc2_id, poi_id
                                            )

                                            # Create metadata for this route
                                            metadata = {
                                                "route_id": route_id,
                                                "start_id": loc2_id,
                                                "destination_id": poi_id,
                                                "base_route_id": base_route_id,
                                                "fraction": 1.0
                                                - fraction,  # Inverse fraction for other direction
                                                "detour_distance_km": detour_distance_km,
                                                "route_alternative": route_idx,
                                                "on_route": True,
                                                "is_reverse": is_true_reverse,
                                            }

                                            # Write metadata to file
                                            f_metadata.write(
                                                json.dumps(metadata) + "\n"
                                            )
                                            routes_metadata_count += 1

                                            # Create symmetric POI to loc route metadata
                                            poi_to_loc2_route_id = generate_route_id(
                                                poi_id, loc2_id
                                            )

                                            # IMPORTANT FIX: For POI→loc2 routes, use the original loc1→loc2 route as base
                                            # Get the corresponding loc1→loc2 route based on route_idx
                                            orig_route_base_id = route_alternatives[
                                                min(
                                                    route_idx,
                                                    len(route_alternatives) - 1,
                                                )
                                            ]["route_id"]

                                            poi_to_loc2_metadata = {
                                                "route_id": poi_to_loc2_route_id,
                                                "start_id": poi_id,
                                                "destination_id": loc2_id,
                                                "base_route_id": orig_route_base_id,  # Use loc1→loc2 route for POI→loc2
                                                "fraction": 1.0 - fraction,
                                                "detour_distance_km": detour_distance_km,
                                                "route_alternative": route_idx,
                                                "on_route": True,
                                                "is_reverse": False,  # Not reversed since using the correct direction's base route
                                            }

                                            f_metadata.write(
                                                json.dumps(poi_to_loc2_metadata) + "\n"
                                            )
                                            routes_metadata_count += 1

                                        # Mark as processed after all alternatives
                                        already_processed_routes.add((loc2_id, poi_id))
                                        already_processed_routes.add((poi_id, loc2_id))

                                else:
                                    print(
                                        f"Warning: Could not find location objects for loc1_id={loc1_id} or loc2_id={loc2_id}"
                                    )

                                f_pois_append.write(json.dumps(poi_data) + "\n")

        print(
            f"Phase 2b: Added {pois_along_routes_count} POIs along fastest Loc-Loc routes."
        )
        print(
            f"Phase 2b: Generated metadata for {routes_metadata_count} routes to/from POIs."
        )
        print(f"Total unique POIs after Phase 2b: {len(all_generated_pois)}")
        print(f"Total unique route pairs processed: {len(already_processed_routes)}")

    # --- In Phase 3 and 4, skip routes already generated in Phase 2b ---
    # Add this at the beginning of your Phase 3 loop:

    # --- Generation Phase 3: Location -> POI Routes (Local Routes Pre-Generated) ---
    print("\n--- Phase 3: Generating Local Routes and Metadata for Other POIs ---")
    meta_loc_poi_count = 0
    local_route_count = 0

    # We'll need to know which location each POI corresponds to
    poi_to_corresponding_loc = {
        poi_id: poi_data["corresponding_location_id"]
        for poi_id, poi_data in all_generated_pois.items()
        if "corresponding_location_id" in poi_data
    }

    print(f"Phase 3: Processing Location-POI pairs...")
    with open(routes_metadata_file, "a", encoding="utf-8") as f_metadata, open(
        routes_loc_poi_file, "a", encoding="utf-8"
    ) as f_loc_poi, open(routes_poi_loc_file, "a", encoding="utf-8") as f_poi_loc:

        # Wrap outer loop with tqdm for better progress
        for location in tqdm(locations, desc="Phase 3: Location->POI Routes"):
            loc_id = location["id"]

            for poi_id, poi_data in all_generated_pois.items():
                # Skip if POI is along the route
                if poi_id in poi_ids_along_routes:
                    continue
                # Skip if we already generated this route in Phase 2b
                if (loc_id, poi_id) in already_processed_routes:
                    continue

                # Ensure poi_data has necessary keys for generation
                if not all(
                    k in poi_data
                    for k in ("id", "name", "position", "corresponding_location_id")
                ):
                    continue

                corresponding_loc_id = poi_data["corresponding_location_id"]

                # --- LOCAL ROUTES: FULLY PRE-GENERATE ---
                # If POI corresponds to starting location, generate complete routes
                if corresponding_loc_id == loc_id:
                    poi_position = poi_data["position"]
                    poi_entity = {
                        "id": poi_id,
                        "name": poi_data["name"],
                        "position": poi_position,
                    }

                    # Generate actual routes (2 alternatives) for local travel
                    routes_loc_to_poi = generate_route_alternatives(
                        location,
                        poi_entity,
                        average_speed_kph=50,  # Slower speed for local travel
                        num_alternatives=2,
                    )

                    # Write the generated route(s) to file
                    for route in routes_loc_to_poi:
                        route_to_write = route.copy()
                        route_to_write.pop("name_via_parts", None)  # Clean helper field
                        f_loc_poi.write(json.dumps(route_to_write) + "\n")
                        local_route_count += 1

                    # Generate symmetric routes for POI to location
                    routes_poi_to_loc = create_symmetric_routes(
                        routes_loc_to_poi, poi_id, loc_id
                    )

                    for route in routes_poi_to_loc:
                        route_to_write = route.copy()
                        route_to_write.pop("name_via_parts", None)
                        f_poi_loc.write(json.dumps(route_to_write) + "\n")
                        local_route_count += 1

                    # Mark as processed
                    already_processed_routes.add((loc_id, poi_id))
                    already_processed_routes.add((poi_id, loc_id))
                    continue

                # --- NON-LOCAL ROUTES: USE METADATA ---
                # Find base routes from location to POI's corresponding location
                base_routes = all_loc_loc_routes_details.get(
                    (loc_id, corresponding_loc_id), []
                )

                # Skip if no base routes available
                if not base_routes:
                    continue

                # Generate metadata for all route alternatives
                for route_idx, base_route in enumerate(base_routes):
                    base_route_id = base_route["route_id"]

                    # Generate route IDs
                    route_id = generate_route_id(loc_id, poi_id)
                    reverse_route_id = generate_route_id(poi_id, loc_id)

                    # Create metadata for the route
                    metadata = {
                        "route_id": route_id,
                        "start_id": loc_id,
                        "destination_id": poi_id,
                        "base_route_id": base_route_id,
                        "route_alternative": route_idx,
                    }

                    # Write metadata to file
                    f_metadata.write(json.dumps(metadata) + "\n")
                    meta_loc_poi_count += 1

                    # Look for reverse base route
                    reverse_routes = all_loc_loc_routes_details.get(
                        (corresponding_loc_id, loc_id), []
                    )
                    if reverse_routes and route_idx < len(reverse_routes):
                        reverse_base_id = reverse_routes[route_idx]["route_id"]
                        is_true_reverse = False
                    else:
                        reverse_base_id = base_route_id
                        is_true_reverse = True

                    # Create symmetric POI to loc metadata
                    reverse_metadata = {
                        "route_id": reverse_route_id,
                        "start_id": poi_id,
                        "destination_id": loc_id,
                        "base_route_id": reverse_base_id,
                        "route_alternative": route_idx,
                        "is_reverse": is_true_reverse,
                    }

                    # Write metadata to file
                    f_metadata.write(json.dumps(reverse_metadata) + "\n")
                    meta_loc_poi_count += 1

                # Mark as processed
                already_processed_routes.add((loc_id, poi_id))
                already_processed_routes.add((poi_id, loc_id))

    print(f"Phase 3: Generated {local_route_count} pre-computed local routes")
    print(
        f"Phase 3: Generated metadata for {meta_loc_poi_count} non-local Location<->POI routes"
    )
    print(f"Total unique route pairs processed: {len(already_processed_routes)}")

    # --- Final Summary ---
    print(f"\n--- Generation Summary ---")
    print(f"Locations generated: {len(locations)}")
    print(f"Unique POIs generated: {len(all_generated_pois)}")
    print(f"Location->Location route alternatives: {loc_loc_route_count}")
    print(f"Route metadata entries: {routes_metadata_count + meta_loc_poi_count}")
    print(f"Weather forecasts generated: {len(locations)}")
    # Count IDs collected during generation
    print(
        f"Total Location IDs: {len(LOCATION_IDS)}, POI IDs: {len(POI_IDS)}, Route IDs: {len(ROUTE_IDS)}, Plug IDs: {len(PLUG_IDS)}"
    )
    print("--------------------------")
    print("Normalized JSONL files generated successfully!")
    print(f"Output directory: {nav_output_dir}")  # Point to navigation subdir

    # --- Create Route Index File for Fast Lookup ---
    print("\n--- Creating Route Index for Fast Lookup ---")
    routes_index_file = os.path.join(nav_output_dir, "routes_index.jsonl")
    output_files.append(routes_index_file)

    route_index = []

    # Index loc-loc routes
    print("Indexing location-location routes...")
    for route_data in tqdm(read_jsonl_file(routes_loc_loc_file)):
        route_id = route_data.get("route_id")
        start_id = route_data.get("start_id")
        dest_id = route_data.get("destination_id")
        if route_id and start_id and dest_id:
            route_index.append(
                {
                    "route_id": route_id,
                    "start_id": start_id,
                    "destination_id": dest_id,
                    "type": "loc-loc",
                }
            )

    # Index loc-poi routes
    print("Indexing location-POI routes...")
    for route_data in tqdm(read_jsonl_file(routes_loc_poi_file)):
        route_id = route_data.get("route_id")
        start_id = route_data.get("start_id")
        dest_id = route_data.get("destination_id")
        if route_id and start_id and dest_id:
            route_index.append(
                {
                    "route_id": route_id,
                    "start_id": start_id,
                    "destination_id": dest_id,
                    "type": "loc-poi",
                }
            )

    # Index poi-loc routes
    print("Indexing POI-location routes...")
    for route_data in tqdm(read_jsonl_file(routes_poi_loc_file)):
        route_id = route_data.get("route_id")
        start_id = route_data.get("start_id")
        dest_id = route_data.get("destination_id")
        if route_id and start_id and dest_id:
            route_index.append(
                {
                    "route_id": route_id,
                    "start_id": start_id,
                    "destination_id": dest_id,
                    "type": "poi-loc",
                }
            )

    # Index metadata routes
    print("Indexing metadata routes...")
    for meta_data in tqdm(read_jsonl_file(routes_metadata_file)):
        route_id = meta_data.get("route_id")
        start_id = meta_data.get("start_id")
        dest_id = meta_data.get("destination_id")
        if route_id and start_id and dest_id:
            route_index.append(
                {
                    "route_id": route_id,
                    "start_id": start_id,
                    "destination_id": dest_id,
                    "type": "metadata",
                }
            )

    # Write the index to file
    with open(routes_index_file, "w", encoding="utf-8") as f_index:
        for entry in route_index:
            f_index.write(json.dumps(entry) + "\n")

    print(f"Created route index with {len(route_index)} entries")


# --- Ensure script execution block ---
if __name__ == "__main__":
    try:
        # Make sure necessary helper functions and classes are defined or imported above
        main()
    except Exception as e:
        print(f"\n--- An error occurred during script execution ---")
        import traceback

        traceback.print_exc()
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("-------------------------------------------------")
