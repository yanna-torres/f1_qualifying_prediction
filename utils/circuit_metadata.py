"""
utils/circuit_metadata.py
====================
Manual lookup table of circuit characteristics, keyed by the original
Circuit name as it appears in qualifying_dataset_wide_with_fp.csv
(before Label Encoding is applied in data.py).

Used to enrich the dataset with circuit-type features that the
Label-Encoded `Circuit` column cannot express on its own (an
arbitrary integer ID carries no information about track character,
so a model has no way to learn e.g. "low-speed street circuits
compress the grid" unless that information is given explicitly).

Categorical attributes
-----------------------
circuit_layout   : "street" | "permanent" | "hybrid"
                    hybrid = permanent circuit built partly on public
                    roads / temporary sections (e.g. Albert Park retains
                    some street-circuit character after 2022 changes;
                    Jeddah is built on public roads but run as a
                    permanent-style layout).
circuit_speed     : "low" | "medium" | "high"
                    Based on average lap speed / proportion of
                    high-speed corners, not top speed alone.
circuit_character : "technical" | "balanced" | "power"
                    "technical" = corner-dominant, low top-speed demand
                    "power"     = long straights, high top-speed demand
                    "balanced"  = mix of both

Numeric attributes
-------------------
track_length_km    : official circuit length in kilometres
num_corners         : official corner count
elevation_change_m  : approximate elevation change in metres (proxy
                      for how much the layout deviates from flat)

Sources: FIA circuit specifications / publicly listed track data as
of the 2022-2025 seasons. Where a circuit's layout changed within the
period (e.g. Spa 2022 vs Spa pre-2022), the most recent layout used
in this dataset's seasons is used.
"""

CIRCUIT_METADATA = {
    "BAHRAIN GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 5.412,
        "num_corners": 15,
        "elevation_change_m": 15,
    },
    "SAUDI ARABIAN GRAND PRIX": {
        "circuit_layout": "street",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 6.174,
        "num_corners": 27,
        "elevation_change_m": 5,
    },
    "AUSTRALIAN GRAND PRIX": {
        "circuit_layout": "hybrid",
        "circuit_speed": "high",
        "circuit_character": "balanced",
        "track_length_km": 5.278,
        "num_corners": 14,
        "elevation_change_m": 10,
    },
    "AZERBAIJAN GRAND PRIX": {
        "circuit_layout": "street",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 6.003,
        "num_corners": 20,
        "elevation_change_m": 15,
    },
    "MIAMI GRAND PRIX": {
        "circuit_layout": "hybrid",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 5.412,
        "num_corners": 19,
        "elevation_change_m": 5,
    },
    "GRAND PRIX DE MONACO": {
        "circuit_layout": "street",
        "circuit_speed": "low",
        "circuit_character": "technical",
        "track_length_km": 3.337,
        "num_corners": 19,
        "elevation_change_m": 42,
    },
    "GRAND PRIX DU CANADA": {
        "circuit_layout": "hybrid",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 4.361,
        "num_corners": 14,
        "elevation_change_m": 5,
    },
    "GRAN PREMIO DE ESPAÑA": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 4.657,
        "num_corners": 14,
        "elevation_change_m": 30,
    },
    "AUSTRIAN GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 4.318,
        "num_corners": 10,
        "elevation_change_m": 65,
    },
    "BRITISH GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "balanced",
        "track_length_km": 5.891,
        "num_corners": 18,
        "elevation_change_m": 25,
    },
    "HUNGARIAN GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "low",
        "circuit_character": "technical",
        "track_length_km": 4.381,
        "num_corners": 14,
        "elevation_change_m": 35,
    },
    "BELGIAN GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 7.004,
        "num_corners": 19,
        "elevation_change_m": 100,
    },
    "DUTCH GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "technical",
        "track_length_km": 4.259,
        "num_corners": 14,
        "elevation_change_m": 30,
    },
    "GRAN PREMIO D’ITALIA": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 5.793,
        "num_corners": 11,
        "elevation_change_m": 7,
    },
    "SINGAPORE GRAND PRIX": {
        "circuit_layout": "street",
        "circuit_speed": "low",
        "circuit_character": "technical",
        "track_length_km": 4.940,
        "num_corners": 19,
        "elevation_change_m": 5,
    },
    "UNITED STATES GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 5.513,
        "num_corners": 20,
        "elevation_change_m": 41,
    },
    "GRAN PREMIO DE LA CIUDAD DE MÉXICO": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 4.304,
        "num_corners": 17,
        "elevation_change_m": 8,
    },
    "GRANDE PRÊMIO DE SÃO PAULO": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 4.309,
        "num_corners": 15,
        "elevation_change_m": 43,
    },
    "LAS VEGAS GRAND PRIX": {
        "circuit_layout": "street",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 6.201,
        "num_corners": 17,
        "elevation_change_m": 3,
    },
    "QATAR GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "balanced",
        "track_length_km": 5.380,
        "num_corners": 16,
        "elevation_change_m": 6,
    },
    "ABU DHABI GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 5.281,
        "num_corners": 16,
        "elevation_change_m": 5,
    },
    "JAPANESE GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "technical",
        "track_length_km": 5.807,
        "num_corners": 18,
        "elevation_change_m": 40,
    },
    "CHINESE GRAND PRIX": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "balanced",
        "track_length_km": 5.451,
        "num_corners": 16,
        "elevation_change_m": 10,
    },
    "GRAND PRIX DE FRANCE": {
        "circuit_layout": "permanent",
        "circuit_speed": "high",
        "circuit_character": "power",
        "track_length_km": 5.842,
        "num_corners": 15,
        "elevation_change_m": 30,
    },
    "GRAN PREMIO DEL MADE IN ITALY E DELL'EMILIA-ROMAGNA": {
        "circuit_layout": "permanent",
        "circuit_speed": "medium",
        "circuit_character": "technical",
        "track_length_km": 4.909,
        "num_corners": 19,
        "elevation_change_m": 20,
    },
}


def get_circuit_metadata_df():
    """
    Returns CIRCUIT_METADATA as a DataFrame indexed by Circuit name,
    ready to be merged onto the main dataset via:

        df = df.merge(get_circuit_metadata_df(), on="Circuit", how="left")
    """
    import pandas as pd

    df = pd.DataFrame.from_dict(CIRCUIT_METADATA, orient="index")
    df.index.name = "Circuit"
    return df.reset_index()


if __name__ == "__main__":
    df = get_circuit_metadata_df()
    print(df.to_string())
