import multiprocessing
import os
import re
import subprocess
import tarfile

from tqdm.auto import tqdm

# List of URLs to download
urls = [
    "https://im2markup.yuntiandeng.com/data/im2latex_formulas.norm.lst",
    "https://im2markup.yuntiandeng.com/data/formula_images_processed.tar.gz",
    "https://im2markup.yuntiandeng.com/data/im2latex_train_filter.lst",
    "https://im2markup.yuntiandeng.com/data/im2latex_validate_filter.lst",
    "https://im2markup.yuntiandeng.com/data/im2latex_test_filter.lst",
]

# Directory to save the downloaded files (in the same directory as the script)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_dir = os.path.join(project_dir, "dataset/data")

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)


# Function to download a single file using wget
def download_file(url):
    file_name = os.path.join(download_dir, url.split("/")[-1])

    # Check if the file already exists; if yes, skip downloading
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        print(f"File already exists, skipping download: {file_name}")
        return

    try:
        # Use subprocess to call wget to download the file
        subprocess.run(["wget", url, "-O", file_name], check=True)

        print(f"Downloaded: {file_name}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

    # If it's the tar.gz file, extract it using the tarfile module
    if file_name.endswith(".tar.gz"):
        print(f"Extracting: {file_name}")
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print(f"Extracted: {file_name}")


def download_data():
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=len(urls))

    # Download each file in parallel
    pool.map(download_file, urls)

    # Close the pool
    pool.close()
    pool.join()
    print("All downloads complete.")


def preprocess_latex():
    with open(os.path.join(download_dir, "im2latex_formulas.norm.lst"), "r") as f:
        formulas = [line.rstrip() for line in f.readlines()]

    print(f"Loaded {len(formulas)} formulas")
    print("Preprocessing LaTeX formulas...")
    for i, formula in tqdm(enumerate(formulas), total=len(formulas)):
        # Ref: https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/dataset/preprocessing/preprocess_formulas.py#L64-L65
        # Replace split, align, etc. with aligned
        formula = re.sub(
            r"\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}",
            r"\\begin{aligned}\2\\end{aligned}",
            formula,
            flags=re.S,
        )
        # Replace smallmatrix with matrix
        formula = re.sub(
            r"\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}",
            r"\\begin{matrix}\2\\end{matrix}",
            formula,
            flags=re.S,
        )

        # Replace \vspace{...} and \hspace{...} with an empty string
        formula = re.sub(r"\\vspace\s*(\*)?\s*\{[^}]*\}", "", formula)
        formula = re.sub(r"\\hspace\s*(\*)?\s*\{[^}]*\}", "", formula)
        formulas[i] = formula

    # Save the formulas to a file
    output_file = os.path.join(download_dir, "im2latex_formulas.final.lst")
    with open(output_file, "w") as f:
        f.write("\n".join(formulas))
        f.write("\n")
        print(f"{len(formulas)} formulas saved to {output_file}")


if __name__ == "__main__":
    download_data()
    preprocess_latex()
