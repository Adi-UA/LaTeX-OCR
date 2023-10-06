import json
import multiprocessing
import os
import shutil
import sys
import time

from munch import Munch
from PIL import Image

from pix2tex.cli import LatexOCR

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(project_dir, "dataset/data")


with open(os.path.join(data_dir, "im2latex_test_filter.lst"), "r") as f:
    test_entries = f.readlines()
    test_data = [(entry.split()[0], int(entry.split()[1])) for entry in test_entries]

with open(os.path.join(data_dir, "im2latex_formulas.final.lst"), "r") as f:
    formulae = f.readlines()


def collect_predictions(pid: int, test_data: list, model: LatexOCR):
    results = {}
    errors = {}
    for img_name, line_no in test_data:
        print(f"({pid}) Processing image {img_name}...")
        img_path = os.path.join(data_dir, "formula_images_processed", img_name)
        if os.path.exists(img_path):
            try:
                ground_truth = formulae[line_no]
                img = Image.open(img_path)
                prediction = model(img)
                results[img_name.rstrip(".png")] = (ground_truth, prediction)
            except Exception as e:
                errors[img_name.rstrip(".png")] = str(e)
        else:
            print(f"({pid}) Image {img_path} does not exist. Skipping.")

    return results, errors


if __name__ == "__main__":
    # Load model
    custom_config = custom_arguments = {
        "config": os.path.join(project_dir, "custom/train_tok_config.yaml"),
        "checkpoint": os.path.join(project_dir, "custom_checkpoints/model.pth"),
        "tokenizer": os.path.join(project_dir, "custom/train_tokenizer.json"),
        "no_cuda": True,
        "no_resize": False,
    }
    model = LatexOCR(Munch(custom_config))

    start_time = time.perf_counter()

    # Split test data into n chunks and run each chunk in parallel with Pool
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    chunk_size = len(test_data) // n
    chunks = [
        test_data[i : i + chunk_size] for i in range(0, len(test_data), chunk_size)
    ]
    with multiprocessing.Pool(n) as pool:
        results = pool.starmap(
            collect_predictions, [(i, chunk, model) for i, chunk in enumerate(chunks)]
        )
    # Merge results
    all_results = {}
    all_errors = {}
    for result, error in results:
        all_results.update(result)
        all_errors.update(error)

    # Create results directory if it doesn't exist
    if not os.path.exists(os.path.join(project_dir, "results")):
        os.mkdir(os.path.join(project_dir, "results"))

    results_dir = os.path.join(project_dir, "results")

    # Save results
    with open(os.path.join(results_dir, "im2latex100k_test_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    with open(os.path.join(results_dir, "im2latex100k_test_errors.json"), "w") as f:
        json.dump(all_errors, f, indent=4)

    end_time = time.perf_counter()
    print(f"Finished in {end_time - start_time} seconds.")
