import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
import random
from tqdm import tqdm, trange
from scipy.spatial import distance

from biked_commons.resource_utils import resource_path, split_datasets_path
from biked_commons.transformation import one_hot_encoding


def prepare_bike_bench():
    data = pd.read_csv(resource_path("datasets/raw_datasets/bike_bench_mixed_modality.csv"), index_col=0)
    data_oh = one_hot_encoding.encode_to_continuous(data)

    columns_to_scale = ['Wall thickness Bottom Bracket', 'Wall thickness Top tube',
            'Wall thickness Head tube', 'Wall thickness Down tube',
            'Wall thickness Chain stay', 'Wall thickness Seat stay',
            'Wall thickness Seat tube']
    num = len(columns_to_scale)

    covariance = np.full((num, num), 0.5)
    covariance[np.diag_indices(num)] = 1.0

    n = np.random.multivariate_normal(np.zeros(num), covariance, size = len(data_oh))
    log_normal_samples = np.exp(n)
    log_normal_samples

    #multiply columns_to_scale with log normal samples
    data_subset = data_oh[columns_to_scale].copy()
    new_values = data_subset.values * log_normal_samples * 2.0
    data_oh[columns_to_scale] = new_values
    data[columns_to_scale] = new_values

    #get any rows where any of the column values is more than 25 standard deviations away from the mean
    def drop_outlier_rows(df, threshold=10):
        return df[~((df - df.mean()).abs() > (threshold * df.std())).any(axis=1)]
    
    data_oh = drop_outlier_rows(data_oh, threshold=25)
    print(f"Removed {len(data) - len(data_oh)} outliers from the dataset.")


    data_oh.to_csv(split_datasets_path("bike_bench.csv"))

    data_subset = data.loc[data_oh.index,:]

    #convert column BELTorCHAIN to bool
    data_subset['BELTorCHAIN'] = data_subset['BELTorCHAIN'].astype(bool)
    data_subset.to_csv(split_datasets_path("bike_bench_mixed_modality.csv"))

def prepare_validity():
    df = pd.read_csv(resource_path('datasets/raw_datasets/validity.csv'), index_col=0)
    df = df.reset_index(drop=True)
    subset = df[df['valid'].isin([0, 2])]
    Y = subset['valid']
    X = subset.drop('valid', axis=1)
    Y = Y.replace({0: 0, 2: 1})  # Convert to binary classification (0 and 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    X_train.to_csv(resource_path('datasets/split_datasets/validity_X_train.csv'))
    X_test.to_csv(resource_path('datasets/split_datasets/validity_X_test.csv'))
    Y_train.to_csv(resource_path('datasets/split_datasets/validity_Y_train.csv'))
    Y_test.to_csv(resource_path('datasets/split_datasets/validity_Y_test.csv'))

def prepare_structure():
    df = pd.read_csv(resource_path('datasets/raw_datasets/structure.csv'), index_col=0)
    df = df.reset_index(drop=True)
    df = df.drop(columns=["batch"])

    sim_1_displacements = df[["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp."]].values
    sim_1_abs_displacements = np.abs(sim_1_displacements)
    sim_1_normalized_displacements = sim_1_abs_displacements / np.mean(sim_1_abs_displacements, axis=0)
    sim_1_compliance_score = np.mean(sim_1_normalized_displacements, axis=1)

    sim_2_displacements = df["Sim 2 Bottom Bracket Z Disp."].values
    sim_2_abs_displacements = np.abs(sim_2_displacements)
    sim_2_compliance_score = sim_2_abs_displacements / np.mean(sim_2_displacements)

    sim_3_displacements = df[["Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot."]].values
    sim_3_abs_displacements = np.abs(sim_3_displacements)
    sim_3_normalized_displacements = sim_3_abs_displacements / np.mean(sim_3_abs_displacements, axis=0)
    sim_3_compliance_score = np.mean(sim_3_normalized_displacements, axis=1)

    mass = df["Model Mass"].values
    planar_SF = df["Sim 1 Safety Factor"].values
    eccentric_SF = df["Sim 3 Safety Factor"].values

    Y = np.stack([mass, sim_1_compliance_score, sim_2_compliance_score, sim_3_compliance_score, planar_SF, eccentric_SF], axis=1)
    Y = pd.DataFrame(Y, columns=["Mass", "Planar Compliance", "Transverse Compliance", "Eccentric Compliance", "Planar Safety Factor", "Eccentric Safety Factor"])
    X = df.drop(["Model Mass", "Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot.", "Sim 1 Safety Factor", "Sim 3 Safety Factor"], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train.to_csv(resource_path('datasets/split_datasets/structure_X_train.csv'))
    X_test.to_csv(resource_path('datasets/split_datasets/structure_X_test.csv'))
    Y_train.to_csv(resource_path('datasets/split_datasets/structure_Y_train.csv'))
    Y_test.to_csv(resource_path('datasets/split_datasets/structure_Y_test.csv'))

def prepare_aero():
    df = pd.read_csv(resource_path('datasets/raw_datasets/aero.csv'), index_col=0)
    df = df.reset_index(drop=True)

    Y = df["drag"]
    X = df.drop("drag", axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train.to_csv(resource_path('datasets/split_datasets/aero_X_train.csv'))
    X_test.to_csv(resource_path('datasets/split_datasets/aero_X_test.csv'))
    Y_train.to_csv(resource_path('datasets/split_datasets/aero_Y_train.csv'))
    Y_test.to_csv(resource_path('datasets/split_datasets/aero_Y_test.csv'))

def download_file(file_url, file_path):
    """Downloads a file with a progress bar if it doesn't exist locally."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists

    response = requests.get(file_url, stream=True)
    
    if response.status_code == 200:
        file_size = int(response.headers.get('content-length', 0))  # Get file size if available
        chunk_size = 1024  # 1 KB per chunk

        with open(file_path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(file_path)}",
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"✅ Download complete: {file_path}")
    else:
        print(f"❌ Error downloading {file_path}: {response.status_code}")

def check_download_CLIP_data():
    # File paths and URLs
    x_id = "10541435"
    x_url = f"https://dataverse.harvard.edu/api/access/datafile/{x_id}"
    x_file = resource_path('datasets/split_datasets/CLIP_X_train.csv')

    y_id = "10992683"
    y_url = f"https://dataverse.harvard.edu/api/access/datafile/{y_id}"
    y_file = resource_path('datasets/split_datasets/CLIP_Y_train.npy')

    # Check and download X
    if not os.path.exists(x_file):
        print(f"⚠️  {os.path.basename(x_file)} not found in datasets folder. Performing first-time download from Harvard Dataverse...")
        download_file(x_url, x_file)
    else:
        print(f"✅ {os.path.basename(x_file)} already exists in datasets folder. Skipping download.")

    # Check and download Y
    if not os.path.exists(y_file):
        print(f"⚠️  {os.path.basename(y_file)} not found in datasets folder. Performing first-time download from Harvard Dataverse...")
        download_file(y_url, y_file)
    else:
        print(f"✅ {os.path.basename(y_file)} already exists in datasets folder. Skipping download.")

def prepare_clip():
    check_download_CLIP_data()

def load_named_colors():
    file_path = resource_path('datasets/raw_datasets/colors.txt')
    colors = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, hex_code = line.strip().split('\t')
            name = name.replace(" ", "-")  # Replace spaces with hyphens
            rgb = tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5))  # Convert hex to RGB
            colors[name] = rgb
    return colors

def nearest_named_color(r, g, b, colors):
    target_rgb = (r, g, b)
    nearest_color = None
    min_distance = float('inf')
    
    for name, rgb in colors.items():
        dist = distance.euclidean(target_rgb, rgb)
        if dist < min_distance:
            min_distance = dist
            nearest_color = name
            
    return nearest_color

def generate_wheel_description(df_slice, probabilities):
    # Helper function to determine "a" or "an"
    def get_article(word):
        return "an" if word and word[0].lower() in "aeiou" else "a"
    
    # Wheel adjectives
    wheel_adjectives_1 = ["", "aerodynamic ", "sleek ", "futuristic ", "lightweight ", "outstanding ", "superior "]
    wheel_adjectives_2 = ["", "ultra-efficient ", "race-optimized ", "high-performance ", "precision-engineered "]
    
    # Determine wheel styles
    rim_style_front = max(["spoked", "trispoke", "disc"], key=lambda x: df_slice[f"RIM_STYLE front OHCLASS: {x.upper()}"])
    rim_style_rear = max(["spoked", "trispoke", "disc"], key=lambda x: df_slice[f"RIM_STYLE rear OHCLASS: {x.upper()}"])

    # Probability-based inclusion of spoke descriptor for front wheel if it’s trispoke
    spoke_descriptor_front = ""
    if rim_style_front == "trispoke" and np.random.rand() < probabilities["spoke_count"]:
        num_spokes_front = df_slice["SPOKES composite front"]
        if num_spokes_front == 3:
            spoke_descriptor_front = "tri-spoked "
        elif num_spokes_front == 4:
            spoke_descriptor_front = "quad-spoked "
        elif num_spokes_front == 5:
            spoke_descriptor_front = "penta-spoked "
        elif num_spokes_front == 6:
            spoke_descriptor_front = "hexa-spoked "
        else:
            spoke_descriptor_front = f"{num_spokes_front}-spoked " if num_spokes_front > 0 else ""

    # Construct wheel prefix
    wheel_prefix = str(np.random.choice(wheel_adjectives_1)) + str(np.random.choice(wheel_adjectives_2))
    article = get_article(wheel_prefix)

    # Handle combinations of front and rear wheel styles, omitting spoked-only setups
    if rim_style_front == "spoked" and rim_style_rear == "spoked":
        return None  # No notable feature for default spoked setup

    if rim_style_front == "trispoke" and rim_style_rear == "trispoke":
        return f"{wheel_prefix}{spoke_descriptor_front}composite wheels"

    elif rim_style_front == "disc" and rim_style_rear == "disc":
        return f"{wheel_prefix}disc wheels"

    elif rim_style_front == "trispoke" and rim_style_rear == "disc":
        return f"{article} {wheel_prefix}{spoke_descriptor_front}composite front wheel and disc rear wheel"

    elif rim_style_front == "disc" and rim_style_rear == "trispoke":
        return f"{article} {wheel_prefix}disc front wheel and composite rear wheel"

    elif rim_style_front == "trispoke" and rim_style_rear == "spoked":
        return f"{article} {wheel_prefix}{spoke_descriptor_front}composite front wheel"

    elif rim_style_front == "disc" and rim_style_rear == "spoked":
        return f"{article} {wheel_prefix}disc front wheel"

    elif rim_style_front == "spoked" and rim_style_rear == "trispoke":
        return f"{article} {wheel_prefix}tri-spoked composite rear wheel"

    elif rim_style_front == "spoked" and rim_style_rear == "disc":
        return f"{article} {wheel_prefix}disc rear wheel"

    return None


def generate_bike_name():
    prefixes = [
        "Thunder", "Trail", "Racer", "Falcon", "Speed", "Iron", "Shadow",
        "Lightning", "Peak", "Storm", "Vertex", "Titan", "Bolt", "Sprint",
        "Eagle", "Canyon", "Striker", "Vortex", "Blade", "Hawk", "Summit",
        "Nova", "Maverick", "Stealth", "Meteor", "Echo", "Specter", "Avalanche",
        "Blaze", "Phoenix", "Comet", "Challenger", "Fusion", "Patriot", "Pioneer",
        "Voyager", "Inferno", "Jet", "Rogue", "Cyclone", "Whirlwind", "Zenith",
        "Trek", "Ascend", "Forge", "Thunderbolt", "Flash", "Tempest", "Guardian",
        "Hunter", "Arrow", "Champion", "Mariner", "Navigator", "Titan", "Forge",
        "Aurora", "Voyage", "Sentinel", "Frontier", "Pulse", "Stormrider", "Vanguard",
        "Raider", "Apex", "Rocket", "Force", "Voyageur", "Outlander", "Hero",
        "Trailblazer", "Adventurer", "Wildcat", "Phantom", "Sonic", "Dominator",
        "Nomad", "Bravo", "Ironclad", "Patron", "Pilot", "Harbinger", "Enigma",
        "Saber", "Nebula", "Prophet", "Conqueror", "Magnum", "Tornado", "Blitz",
        "Shifter", "Impulse", "Scorpion", "Interceptor", "Viper"
    ]
    
    suffixes = [
        "X", "Pro", "Elite", "Max", "GT", "Ace", "Force", "Sprint", "Wave",
        "Prime", "Ultra", "Turbo", "Carbon", "XR", "One", "Series", "V", "SL",
        "Climb", "CR", "Trail", "Flight", "Rush", "Pulse", "Edge", "Vision",
        "Spirit", "Flow", "Momentum", "Quest", "Surge", "Adventure", "Impact",
        "Hybrid", "Evolution", "Stealth", "Force", "Neo", "Flex", "Enduro",
        "Storm", "Velocity", "Drive", "All-Terrain", "Dynamic", "Fusion", "Terra",
        "Xtreme", "Rider", "Advance", "Supreme", "Commander", "Trailforce",
        "Raptor", "Master", "Invictus", "Champion", "Elite-X", "Infinity", "Voyager",
        "Xpert", "Precision", "Pioneer", "Torque", "Revolution", "Nova", "Excursion",
        "Pursuit", "Altitude", "Shift", "Virtue", "Momentum", "Altitude-X", "Command",
        "Verve", "Impact-X", "Pinnacle", "Rampage", "Circuit", "Ridge", "Axis"
    ]
    
    # Mathematical number patterns for optional endings
    numbers = []
    numbers += [x * 1000 for x in range(1, 10)]  # X000 pattern (1000, 2000, ..., 9000)
    numbers += [x * 100 for x in range(1, 10)]   # X00 pattern (100, 200, ..., 900)
    numbers += [x * 10 for x in range(10, 100)]  # XX0 pattern (100, 110, ..., 990)
    numbers += [x * 100 + 99 for x in range(1, 10)]  # X99 pattern (199, 299, ..., 999)
    
    # Randomly decide whether to add a number (30% chance)
    include_number = np.random.rand() < 0.3
    
    # Construct the bike name
    bike_name = f"{np.random.choice(prefixes)} {np.random.choice(suffixes)}"
    if include_number:
        bike_name += f" {np.random.choice(numbers)}"
    
    return bike_name

def generate_description(df_slice, colors):
    # Consolidated probabilities for adding each type of feature
    probabilities = {
        "color": 0.9,
        "frame": 0.8,
        "handlebar": 0.8,
        "belt_drive": 0.5,
        "wheels": 0.8,
        "bottle_mounts": 0.8,
        "fenders": 0.7,
        "aerobars": 0.7,  # 40% chance of including aerobars
        "cargo_rack": 0.7,  # 40% chance of including cargo rack
        "spoke_count": 0.0
    }
    
    potential_features = []
    feature_prefixes = ["with", "featuring", "equipped with", "boasting", "designed with", "crafted with"]
    feature_prefixes_2 = ["features", "is equipped with", "boasts", "comes with", "includes", "offers"]

    # Color description generation with 20% probability of being an empty string
    red_rgb = df_slice["FIRST color R_RGB"]
    green_rgb = df_slice["FIRST color G_RGB"]
    blue_rgb = df_slice["FIRST color B_RGB"]
    color_desc = nearest_named_color(red_rgb, green_rgb, blue_rgb, colors) + " "
    if np.random.rand() < 1 - probabilities["color"]:  # Adjust based on probability
        color_desc = ""

    # Fender logic
    front_fender = df_slice["Front Fender include"] >= 0.5
    rear_fender = df_slice["Rear Fender include"] >= 0.5

    if np.random.rand() < probabilities["fenders"]:
        if front_fender and rear_fender:
            potential_features.append("fenders for added protection")
        elif front_fender:
            potential_features.append("a front fender for splatter resistance")
        elif rear_fender:
            potential_features.append("a rear fender for trail comfort")

    # Aerobars logic
    display_aerobars = df_slice["Display AEROBARS"] >= 0.5
    if display_aerobars and np.random.rand() < probabilities["aerobars"]:
        aerobar_descriptions = [
            "a set of aerodynamic bars", 
            "integrated aerobars for optimal performance", 
            "aerobars for time trial efficiency", 
            "racing aerobars for streamlined control",
            "aerobars for triathlon dominance",
            "aerodynamic bars for speed",
        ]
        potential_features.append(np.random.choice(aerobar_descriptions))

    # Cargo rack logic
    display_rack = df_slice["Display RACK"] >= 0.5
    if display_rack and np.random.rand() < probabilities["cargo_rack"]:
        cargo_rack_descriptions = [
            "a cargo rack for extra storage", 
            "a sturdy rear rack for carrying gear", 
            "an integrated cargo rack for commuting", 
            "a versatile rack for added utility"
        ]
        potential_features.append(np.random.choice(cargo_rack_descriptions))

    # Frame material logic with enhanced descriptors
    frame_adjectives = {
        "titanium": ["a lightweight", "an indestructible", "an advanced", "a premium", "a high-end", "a resilient", "a rustproof", "an aerospace-grade", "a sleek"],
        "carbon": ["a high-performance", "an ultra-lightweight", "a cutting-edge", "a race-ready", "a stiff", "a modern", "an aerodynamic", "a competition-grade"],
        "steel": ["a classic", "a strong", "a time-tested", "a durable", "a sturdy", "a retro-inspired", "a resilient", "an enduring"],
        "aluminium": ["a versatile", "a lightweight", "a budget-friendly", "a reliable", "a corrosion-resistant", "a robust", "an affordable", "a sporty"],
        "bamboo": ["an eco-friendly", "a sustainable", "a natural", "a unique", "a shock-absorbing", "a lightweight", "an organic", "a renewable-resource"],
        "other": []  # No adjectives for "other" as we want to exclude it
    }

    # Determine the frame type and add a feature
    if np.random.rand() < probabilities["frame"]:
        frame_type = max(frame_adjectives, key=lambda x: df_slice[f"MATERIAL OHCLASS: {x.upper()}"])
        if frame_type != "other":
            frame_adj = np.random.choice(frame_adjectives[frame_type])
            potential_features.append(f"{frame_adj} {frame_type} frame")

    # Handlebar style logic with expanded descriptors
    handlebar_adjectives = {
        "0": ["classic", "aerodynamic", "road-ready", "versatile", "sleek", "precision-designed", "high-speed", "comfort-optimized"],
        "1": ["rugged", "durable", "off-road", "adventure-ready", "wide", "shock-absorbing", "trail-proven", "versatile"],
        "2": ["aggressive", "urban", "streamlined", "race-oriented", "ergonomic", "city-friendly", "fast-handling", "sleek-profiled"]
    }

    handlebar_names = {
        "0": ["drop handlebars", "road handlebars", "curved bars", "drop bars", "race bars"],
        "1": ["MTB handlebars", "mountain bike bars", "flat bars", "trail bars", "mountain bars"],
        "2": ["bullhorn handlebars", "urban bullhorns", "racing bullhorns", "bullhorns", "track bullhorns"]
    }

    if np.random.rand() < probabilities["handlebar"]:
        handlebar_type = max(["0", "1", "2"], key=lambda x: df_slice[f"Handlebar style OHCLASS: {x.upper()}"])
        handlebar_adj = np.random.choice(handlebar_adjectives[handlebar_type])
        handlebar_name = np.random.choice(handlebar_names[handlebar_type])
        potential_features.append(f"{handlebar_adj} {handlebar_name}")

    # Belt drive logic with more detailed descriptions
    if df_slice["BELTorCHAIN OHCLASS: 0"] > df_slice["BELTorCHAIN OHCLASS: 1"] and np.random.rand() < probabilities["belt_drive"]:
        belt_drive_descriptions = [
            "a belt drive system", 
            "an advanced belt drive setup", 
            "a low-maintenance belt drive", 
            "a smooth and quiet belt drive system", 
            "a modern belt drive configuration", 
            "a hassle-free belt drive system", 
            "a clean and efficient belt drive",
            "a precision-engineered belt drive"
        ]
        potential_features.append(np.random.choice(belt_drive_descriptions))

    # Wheel style logic with expanded descriptors
    potential_wheel_feature = generate_wheel_description(df_slice, probabilities)
    if potential_wheel_feature:
        potential_features.append(potential_wheel_feature)



    # Bottle holder logic with expanded postfix options
    if np.random.rand() < probabilities["bottle_mounts"]:
        seattube_bottle = df_slice["bottle SEATTUBE0 show OHCLASS: False"] < df_slice["bottle SEATTUBE0 show OHCLASS: True"]
        downtube_bottle = df_slice["bottle DOWNTUBE0 show OHCLASS: False"] < df_slice["bottle DOWNTUBE0 show OHCLASS: True"]
        bottle_postfix_options = [
            "", 
            " for quick and easy hydration on the go", 
            " to stay hydrated on your ride", 
            " for easy access to water", 
            " for your convenience", 
            " for easy hydration", 
            " for quick and easy access to water",
            " to keep you refreshed during long rides",
            " to make hydration seamless and simple",
            " designed for effortless water access",
            " so you never miss a sip",
            " for those hot, sunny rides"
        ]
        bottle_postfix = np.random.choice(bottle_postfix_options)

        if seattube_bottle and downtube_bottle:
            potential_features.append(f"dual bottle mounts on the seat tube and down tube{bottle_postfix}")
        elif seattube_bottle:
            potential_features.append(f"a seat tube-mounted bottle holder{bottle_postfix}")
        elif downtube_bottle:
            potential_features.append(f"a down tube-mounted bottle holder{bottle_postfix}")

    fallback_features = [
        "a durable build",
        "precision engineering",
        "an ergonomic design",
        "superior craftsmanship",
        "a sleek and modern finish",
        "a versatile frame",
        "rugged construction",
        "advanced components",
        "high-quality materials",
        "a smooth ride experience",
        "a lightweight profile",
        "a unique aesthetic",
        "stability and comfort",
        "state-of-the-art components",
        "a high-tech build",
        "optimized handling",
        "cutting-edge technology",
        "a user-friendly design",
        "superior balance"
    ]

    if not potential_features:
        potential_features.append(np.random.choice(fallback_features))

    if potential_features:
        random.shuffle(potential_features)

    # Generating the bike description with the collected features
    bike_name = generate_bike_name()
    bike_intro = f"the {color_desc}{bike_name}".strip()

    if len(potential_features) > 2:
        feature_list = ', '.join(potential_features[:-1]) + ', and ' + potential_features[-1]
    elif len(potential_features) == 2:
        feature_list = ' and '.join(potential_features)
    elif potential_features:
        feature_list = potential_features[0]

    prefixing = np.random.choice(feature_prefixes)
    prefixes = np.random.choice(feature_prefixes_2)
    sentence_template = np.random.choice([
        f"{bike_intro} {prefixes} {feature_list}.",
        f"{bike_intro} {prefixes} {feature_list} and is the solution to your cycling needs.",
        f"{bike_intro} {prefixes} {feature_list}, ensuring top-tier performance on every ride.",
        f"{bike_intro} {prefixes} {feature_list} for unmatched cycling performance.",
        f"{prefixing} {feature_list}, {bike_intro} is the ultimate machine for your next ride.",
        f"{prefixing} {feature_list}, {bike_intro} is ready to take on any challenge.",
        f"{prefixing} {feature_list}, {bike_intro} sets a new standard in cycling.",
        f"Unleash your inner power with {bike_intro}, {prefixing} {feature_list}.",
        f"Get ready for your best ride yet with {bike_intro}, {prefixing} {feature_list}.",
        f"{prefixing} {feature_list}, {bike_intro} promises unparalleled performance.",
        f"Experience excellence with {bike_intro}, {prefixing} {feature_list}.",
        f"Take your rides to the next level with {bike_intro}, {prefixing} {feature_list}.",
        f"{bike_intro}, {prefixing} {feature_list}, is built for those who demand the best.",
        f"Crafted for performance, {bike_intro} {prefixes} {feature_list} to meet your cycling needs.",
        f"Engineered for your adventures, {bike_intro} {prefixes} {feature_list}.",
        f"Redefine your cycling experience with {bike_intro}, {prefixing} {feature_list}.",
        f"Elevate your journey with {bike_intro}, {prefixing} {feature_list}.",
        f"Perfect for enthusiasts, {bike_intro} {prefixes} {feature_list}.",
        f"Reach new heights with {bike_intro}, {prefixing} {feature_list}.",
        f"Designed to impress, {bike_intro} {prefixes} {feature_list} for all your adventures.",
        f"Discover the freedom of cycling with {bike_intro}, {prefixing} {feature_list}.",
        f"{bike_intro} {prefixes} {feature_list}, perfect for those who demand quality and style.",
        f"{prefixing} {feature_list}, {bike_intro} will redefine your cycling experience.",
        f"Push the limits of your ride with {bike_intro}, {prefixing} {feature_list}.",
        f"Transform your rides with {bike_intro}, {prefixing} {feature_list}.",
        f"Built for champions, {bike_intro} {prefixes} {feature_list} for peak performance.",
        f"{bike_intro} {prefixes} {feature_list}, combining innovation and reliability.",
        f"{prefixing} {feature_list}, {bike_intro} delivers unparalleled comfort and control.",
        f"{prefixing} {feature_list}, {bike_intro} turns every ride into an adventure.",
        f"{bike_intro} {prefixes} {feature_list} to take your cycling further.",
        f"{bike_intro} {prefixes} {feature_list} to ensure an unforgettable ride.",
        f"{bike_intro}, {prefixing} {feature_list}, is crafted to exceed your expectations.",
        f"Enjoy every mile with {bike_intro}, {prefixing} {feature_list} for superior handling.",
        f"{prefixing} {feature_list}, {bike_intro} stands out from the competition.",
        f"Explore new trails with {bike_intro}, {prefixing} {feature_list} for unmatched durability.",
        f"{bike_intro} {prefixes} {feature_list}, making each ride smoother and more enjoyable.",
        f"Conquer any road with {bike_intro}, {prefixing} {feature_list} for exceptional performance.",
        f"Embrace the road ahead with {bike_intro}, {prefixing} {feature_list}.",
        f"{prefixing} {feature_list}, {bike_intro} is the perfect companion for your rides.",
        f"Experience the thrill of the ride with {bike_intro}, {prefixing} {feature_list}.",
        f"Built for those who dare, {bike_intro} {prefixes} {feature_list}.",
        f"Feel the power of the road with {bike_intro}, {prefixing} {feature_list}.",
        f"Take on any challenge with {bike_intro}, {prefixing} {feature_list}.",
        f"{bike_intro} {prefixes} {feature_list}, ready to make every ride memorable.",
        f"Push your boundaries with {bike_intro}, {prefixing} {feature_list}.",
        f"{bike_intro} {prefixes} {feature_list} to support your cycling ambitions."
    ])

    # Ensure the first character is capitalized if needed
    if sentence_template[0].islower():
        sentence_template = sentence_template[0].upper() + sentence_template[1:]

    return sentence_template

def generate_descriptions(df):
    colors = load_named_colors()
    descriptions = []
    for i in trange(len(df)):
        df_slice = df.iloc[i]
        description = generate_description(df_slice, colors)
        descriptions.append(description)
    return descriptions

def prepare_text_descriptions():

    data = pd.read_csv(split_datasets_path("CLIP_X_test.csv"), index_col=0)
    descriptions = generate_descriptions(data)
    with open(split_datasets_path("text_descriptions_test.txt"), "w") as f:
        for desc in tqdm(descriptions):
            f.write(desc + "\n")

    data = pd.read_csv(split_datasets_path("CLIP_X_train.csv"), index_col=0)
    #sample 100k
    data = data.sample(100000)
    descriptions = generate_descriptions(data)
    with open(split_datasets_path("text_descriptions_train.txt"), "w") as f:
        for desc in tqdm(descriptions):
            f.write(desc + "\n")

