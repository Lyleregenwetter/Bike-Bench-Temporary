import logging
import os.path
import time
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from sdv.single_table import CTGANSynthesizer

from typing import Dict
import sys
import pandas as pd
import cairosvg

import os
from typing import Dict
from PIL import Image
import numpy as np
import torch
import shutil
import uuid



from biked_commons.rendering.rendering import RenderingEngine, FILE_BUILDER
from biked_commons.data_loading import data_loading
from biked_commons.resource_utils import resource_path, STANDARD_BIKE_RESOURCE, models_and_scalers_path
from biked_commons.embedding.clip_embedding_calculator import ClipEmbeddingCalculator
from biked_commons.transformation.one_hot_encoding import encode_to_continuous
from biked_commons.validation.base_validation_function import construct_tensor_validator
from biked_commons.validation.clip_validation_functions import CLIPS_VALIDATIONS



def read_standard_xml():
    with open(STANDARD_BIKE_RESOURCE, "r") as file:
        return file.read()

standard_bike_xml = read_standard_xml()

def get_bike_bench_records_with_id(num) -> Dict[str, dict]:
    """
    Return records in a dictionary of the form {
    (record_id: str) : (record: dict)
    }
    """
    data_train = data_loading.load_bike_bench_train()
    data_test = data_loading.load_bike_bench_test()
    data = pd.concat([data_train, data_test], axis=0)
    if num is None:
        num = len(data)
    else:
        data = data.iloc[:num, :]

    return {str(record_id): record for record_id, record in zip(data.index.tolist(), data.to_dict(orient="records"))}

from biked_commons.validation.base_validation_function import construct_tensor_validator, construct_dataframe_validator

def sample_n(n=4096):
    save_path = models_and_scalers_path("CTGAN.pkl")
    synthesizer = CTGANSynthesizer.load(filepath=save_path)
    sample_datapoint = synthesizer.sample(num_rows=1)
    sample_datapoint_oh = encode_to_continuous(sample_datapoint)
    COLUMN_NAMES = list(sample_datapoint_oh.columns)
    tensor_validator, validation_names = construct_tensor_validator(CLIPS_VALIDATIONS, COLUMN_NAMES)

    all_valid_samples = None
    while True:
        synthetic_collapsed = synthesizer.sample(num_rows=10000)
        samples_oh = encode_to_continuous(synthetic_collapsed)
        samples_oh_tens = torch.tensor(samples_oh.values, dtype=torch.float32)

        validity = tensor_validator(samples_oh_tens)

        valid = torch.all(validity<=0, dim=1)
        valid_subset = samples_oh_tens[valid, :]
        if all_valid_samples is None:
            all_valid_samples = valid_subset
        else:
            all_valid_samples = torch.cat((all_valid_samples, valid_subset), dim=0)
        if all_valid_samples.shape[0] >= n:
            break
    all_valid_samples = all_valid_samples[:n, :]
    samples_df = pd.DataFrame(all_valid_samples.numpy(), columns=COLUMN_NAMES)
    return samples_df

def sample_save_n_records(save_path, n=4096):
    data = sample_n(n)
    #make the data indices random keys using uuid
    random_keys = [str(uuid.uuid4()) for _ in range(len(data))]
    data.index = random_keys
    #save csv to save_path
    data.to_csv(save_path)
    return {str(record_id): record for record_id, record in zip(data.index.tolist(), data.to_dict(orient="records"))}

def bike_to_xml(save_path: str, record_id: str, record: dict):
    try:
        file_path = os.path.join(save_path, f"{record_id}.xml")
        with open(file_path, "w") as file:
            xml_data = FILE_BUILDER.build_cad_from_clip(record, standard_bike_xml, False)
            file.write(xml_data)
    except Exception as e:
        print(f"Failed with exception {e}")


def bikes_to_xmls(records_with_id: Dict[str, dict],
                   process_pool_workers: int,
                   save_dir: str
                   ):
    executor = ProcessPoolExecutor(max_workers=process_pool_workers)
    os.makedirs(save_dir, exist_ok=True)
    for record_id, record in records_with_id.items():
        executor.submit(bike_to_xml, save_dir, record_id, record)
    executor.shutdown()  # waits for all submitted tasks to finish

def xmls_to_svgs(
        thread_pool_workers: int,
        records_with_id: Dict[str, dict],
        xml_dir: str,
        svg_dir: str,
        rendering_engine: RenderingEngine
):
    executor = ThreadPoolExecutor(max_workers=thread_pool_workers)

    os.makedirs(svg_dir, exist_ok=True)
    def xml_to_svg(xml: str):
        try:
            xml_path = os.path.join(xml_dir, f"{xml}.xml")
            with open(xml_path, "r") as xml_file:
                # print("Sending request to server...")
                read_file = xml_file.read()
                # print("Read file...")
                rendering_result = rendering_engine.render_xml(read_file)
                # print("Rendering result received from server...")
                image_path = os.path.join(svg_dir, f"{xml}.svg")
                with open(image_path, "wb") as image_file:
                    image_file.write(rendering_result.image_bytes)
                return True
        except Exception as e:
            print(f"Rendering failed: {e}")
            return False, e

    for record_id, _ in records_with_id.items():
        executor.submit(xml_to_svg, record_id)

    executor.shutdown()


def svg_to_png(record_id: str, svg_dir: str, png_dir: str):
        try:
            svg_file = os.path.join(svg_dir, f"{record_id}.svg")
            png_file = os.path.join(png_dir, f"{record_id}.png")
            cairosvg.svg2png(url=svg_file, write_to=png_file)
        except Exception as e:
            print(f"Failed to convert {record_id} with exception {e}")

def svgs_to_pngs(process_pool_workers: int,
               records_with_id: Dict[str, dict],
                svg_dir: str,
               png_dir: str):
    executor = ProcessPoolExecutor(max_workers=process_pool_workers)
    os.makedirs(png_dir, exist_ok=True)
    for record_id, _ in records_with_id.items():
        executor.submit(svg_to_png, record_id, svg_dir, png_dir)
    executor.shutdown()  # waits for all submitted tasks to finish
    
#load all pngs and stack up
def load_pngs(png_dir: str,
              records_with_id: Dict[str, dict]) -> torch.Tensor:
    """
    Scans png_dir for files named <record_id>.png (in the order of records_with_id.keys()),
    loads each as an RGB image, converts to a float tensor in [0,1],
    and stacks them into a tensor of shape (N, 3, H, W).
    """
    imgs = []
    names = []
    for record_id in records_with_id:
        path = os.path.join(png_dir, f"{record_id}.png")
        if not os.path.exists(path):
            # optionally warn, or collect missing IDs
            print(f"Warning: {path} not found; skipping.")
            continue

        with Image.open(path) as img:
            img = img.convert("RGB")                   # ensure 3 channels
            arr = np.array(img)                        # H x W x 3, uint8
            tensor = torch.from_numpy(arr)             # H x W x 3
            tensor = tensor.permute(2, 0, 1)            # 3 x H x W
            tensor = tensor.float().div(255.0)          # scale to [0,1]
            imgs.append(tensor)
            names.append(record_id)

    if not imgs:
        # return an empty tensor if nothing was loaded
        return torch.empty((0, 3, 0, 0))

    return torch.stack(imgs, dim=0), names # N x 3 x H x W

def embed_pngs(
    png_dir: str,
    records_with_id: Dict[str, dict],
    batch_size: int = 32,
    emb_file: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_embedder = ClipEmbeddingCalculator(batch_size=batch_size, device=device)

    all_embs = []
    all_names = []
    ids = list(records_with_id)

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        subset = {rid: records_with_id[rid] for rid in batch_ids}
        imgs, names = load_pngs(png_dir, subset)
        if not names:
            continue

        with torch.no_grad():
            emb = clip_embedder.embed_images(imgs).cpu()
        all_embs.append(emb)
        all_names.extend(names)

    embs = torch.cat(all_embs, dim=0).numpy()
    df = pd.DataFrame(embs, index=all_names)
    df.to_csv(emb_file or "embeddings.csv")


def process_rendering_stack(records, xml_dir: str, svg_dir: str, png_dir: str, emb_file: str, rendering_engine: RenderingEngine, process_pool_workers: int, thread_pool_workers: int):
    bikes_to_xmls(records, process_pool_workers, xml_dir)
    print("XMLs created")

    xmls_to_svgs(
        thread_pool_workers=thread_pool_workers,
        records_with_id=records,
        xml_dir=xml_dir,
        svg_dir=svg_dir,
        rendering_engine=rendering_engine,
    )
    print("SVGs created")
    svgs_to_pngs(process_pool_workers=process_pool_workers, records_with_id=records, svg_dir=svg_dir, png_dir=png_dir)
    print("PNGs created")
    embed_pngs(png_dir=png_dir, records_with_id=records, batch_size=32, emb_file=emb_file)
    print("Embeddings created")
    shutil.make_archive(xml_dir, 'zip', xml_dir)
    shutil.make_archive(svg_dir, 'zip', svg_dir)
    shutil.make_archive(png_dir, 'zip', png_dir)
    print("Zipped all directories")
    shutil.rmtree(xml_dir)
    shutil.rmtree(svg_dir)
    shutil.rmtree(png_dir)
    print("Removed all directories")