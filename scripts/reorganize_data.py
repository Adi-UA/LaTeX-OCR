import multiprocessing
import os
import shutil

from tqdm.auto import tqdm


def read_file_lines(file_path):
    with open(file_path, "r") as f:
        return [line.rstrip() for line in f]


def copy_images_and_create_list(src_image_paths, dest_dir, equations_file, equations):
    """Copy images to dest_dir and create a list of equations in equations_file. The images are renamed to 0.png, 1.png, etc.
    The image's corresponding equation is at the same line number in equations file.

    Args:
        src_image_paths (list): List of source image paths.
        dest_dir (str): Destination directory.
        equations_file (str): File to write equations to.
        equations (list): List of equations. The image's corresponding equation must be at the same index in this list.

    Use only absolute paths for src_image_paths, dest_dir and equations_file.
    """
    os.mkdir(dest_dir)

    print(f"Copying images to {dest_dir}...")
    for i, src_img in tqdm(enumerate(src_image_paths), total=len(src_image_paths)):
        shutil.copy(src_img, os.path.join(dest_dir, f"{i}.png"))
    print(f"Copied {len(src_image_paths)} images to {dest_dir}.")

    print(f"Creating {equations_file}...")
    with open(equations_file, "w") as f:
        f.write("\n".join(equations))
    print(f"Created {equations_file}.")


if __name__ == "__main__":
    # Get the paths to the images and equations
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "dataset/data")
    images_dir = os.path.join(data_dir, "formula_images_processed")
    formulae_file = os.path.join(data_dir, "im2latex_formulas.final.lst")
    train_file = os.path.join(data_dir, "im2latex_train_filter.lst")
    val_file = os.path.join(data_dir, "im2latex_validate_filter.lst")

    # Read the formulae, train and val files
    formulas = read_file_lines(formulae_file)
    train_info = read_file_lines(train_file)
    val_info = read_file_lines(val_file)

    print("Collecting train and val data...")
    # Get the corresponding images and equations for train
    train_formula_images, train_formula_line_nos = [
        os.path.join(images_dir, line.split()[0]) for line in train_info
    ], [int(line.split()[1]) for line in train_info]
    train_formulas = [formulas[line_no] for line_no in train_formula_line_nos]

    # Get the corresponding images and equations for val
    val_formula_images, val_formula_line_nos = [
        os.path.join(images_dir, line.split()[0]) for line in val_info
    ], [int(line.split()[1]) for line in val_info]
    val_formulas = [formulas[line_no] for line_no in val_formula_line_nos]

    # Remove train and val directories if they exist
    print("Removing existing train and val directories from previous runs...")
    shutil.rmtree(os.path.join(data_dir, "train"), ignore_errors=True)
    shutil.rmtree(os.path.join(data_dir, "val"), ignore_errors=True)

    print("Copying images and creating train and val lists...")
    # Copy images and create list for train and val in parallel
    with multiprocessing.Pool(2) as pool:
        pool.starmap(
            copy_images_and_create_list,
            [
                (
                    train_formula_images,
                    os.path.join(data_dir, "train"),
                    os.path.join(data_dir, "train.lst"),
                    train_formulas,
                ),
                (
                    val_formula_images,
                    os.path.join(data_dir, "val"),
                    os.path.join(data_dir, "val.lst"),
                    val_formulas,
                ),
            ],
        )
    print("Done.")
