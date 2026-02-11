import csv
import json
import os


def get_population(city_row_or_dict):
    """Safely get population as an integer from a city data dictionary or CSV row."""
    pop_str = city_row_or_dict.get("population", "")
    if not pop_str or not pop_str.isdigit():  # Check if empty or not a number
        pop_str = city_row_or_dict.get(
            "population_proper", ""
        )  # Fallback to population_proper

    if pop_str and pop_str.isdigit():  # Final check before converting
        return int(pop_str)
    return 0  # Default to 0 if no valid population found


def create_europe_cities_json(countries_to_exclude_by_car=None):
    if countries_to_exclude_by_car is None:
        countries_to_exclude_by_car = set()
    else:
        # Ensure ISO codes are uppercase for consistent comparison
        countries_to_exclude_by_car = set(
            c.upper() for c in countries_to_exclude_by_car
        )

    print("--- Script execution started ---")

    base_dir = "car_bench/envs/car_voice_assistant/mock_data/navigation"
    de_cities_path = os.path.join(base_dir, "de_cities.jsonc")
    world_cities_path = os.path.join(base_dir, "worldcities.csv")
    output_path = os.path.join(base_dir, "europe_cities.jsonc")

    print(f"Using German cities file: {de_cities_path}")
    print(f"Using world cities file: {world_cities_path}")
    print(f"Output file will be: {output_path}")
    if countries_to_exclude_by_car:
        print(
            f"Excluding countries (not reachable by car): {sorted(list(countries_to_exclude_by_car))}"
        )

    european_iso2_codes = {
        "AL",
        "AD",
        "AM",
        "AT",
        "AZ",
        "BY",
        "BE",
        "BA",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "GE",
        "GR",
        "HU",
        "IS",
        "IE",
        "IT",
        "KZ",
        "XK",
        "LV",
        "LI",
        "LT",
        "LU",
        "MT",
        "MD",
        "MC",
        "ME",
        "NL",
        "MK",
        "NO",
        "PL",
        "PT",
        "RO",
        "RU",
        "SM",
        "RS",
        "SK",
        "SI",
        "ES",
        "SE",
        "CH",
        "TR",
        "UA",
        "GB",
    }

    german_cities = []
    if not os.path.exists(de_cities_path):
        print(f"FATAL ERROR: German cities file not found at '{de_cities_path}'")
        return
    try:
        with open(de_cities_path, "r", encoding="utf-8") as f:
            content = "".join(line for line in f if not line.strip().startswith("//"))
            german_cities = json.loads(content)
        print(
            f"Successfully loaded {len(german_cities)} German cities from '{de_cities_path}'."
        )
    except Exception as e:
        print(f"FATAL ERROR: Could not read/parse German cities file. Error: {e}")
        return

    all_europe_cities = list(german_cities)
    german_city_names_lower = {city["city"].lower() for city in german_cities}
    print(f"Initialized all_europe_cities with {len(all_europe_cities)} German cities.")

    if not os.path.exists(world_cities_path):
        print(f"FATAL ERROR: World cities CSV file not found at '{world_cities_path}'")
        return

    try:
        processed_rows = 0
        # Store cities by country first { "ISO2": [city_dict1, city_dict2], ... }
        european_cities_by_country = {}

        with open(world_cities_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_rows += 1
                country_iso2 = row.get("iso2")
                city_name = row.get("city")

                if not city_name:  # Skip rows with no city name
                    continue

                if (
                    country_iso2
                    and country_iso2 in european_iso2_codes
                    and country_iso2 != "DE"
                ):
                    if country_iso2 in countries_to_exclude_by_car:
                        continue  # Skip this country if it's in the exclusion list

                    current_population = get_population(row)
                    city_data = {
                        "city": city_name,
                        "lat": row.get("lat", ""),
                        "lng": row.get("lng", ""),
                        "country": row.get("country", ""),
                        "iso2": country_iso2,
                        "admin_name": row.get("admin_name", ""),
                        "capital": row.get("capital", ""),
                        # Store population as string, consistent with original de_cities format
                        # get_population was used for sorting, but final storage is string.
                        "population": (
                            str(row.get("population", ""))
                            if row.get("population", "").isdigit()
                            else str(get_population(row))
                        ),
                        "population_proper": str(
                            row.get("population_proper", "")
                        ),  # Keep original population_proper as string
                    }

                    if country_iso2 not in european_cities_by_country:
                        european_cities_by_country[country_iso2] = []
                    european_cities_by_country[country_iso2].append(city_data)

        print(
            f"Processed {processed_rows} rows from '{world_cities_path}'. Grouped cities for non-German European countries."
        )

        added_count = 0
        for country_iso2, cities_in_country in european_cities_by_country.items():
            # Sort cities by population (descending) using the helper function
            cities_in_country.sort(key=lambda city: get_population(city), reverse=True)

            for city_to_add in cities_in_country[:2]:  # Take top 2
                # Ensure the city (by name) isn't already in the German list
                # This check is primarily for safety, as German cities should have "DE" ISO2
                if city_to_add["city"].lower() not in german_city_names_lower:
                    all_europe_cities.append(city_to_add)
                    added_count += 1

        print(
            f"Added {added_count} top cities from other European countries (max 2 per country)."
        )

    except Exception as e:
        print(
            f"FATAL ERROR: An error occurred while processing '{world_cities_path}'. Error: {e}"
        )
        return

    try:
        os.makedirs(base_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(all_europe_cities, outfile, indent=2, ensure_ascii=False)
        print(
            f"--- Successfully created '{output_path}' with {len(all_europe_cities)} total cities. ---"
        )
    except Exception as e:
        print(f"FATAL ERROR: Could not write to '{output_path}'. Reason: {e}")


if __name__ == "__main__":
    # Example: To exclude Malta, Cyprus, and Iceland (if not reachable by car)
    # list_of_countries_to_exclude = ['MT', 'CY', 'IS']
    # create_europe_cities_json(countries_to_exclude_by_car=list_of_countries_to_exclude)

    # Default run without car exclusion list
    create_europe_cities_json()
