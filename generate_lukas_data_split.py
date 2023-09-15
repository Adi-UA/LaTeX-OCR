import os
import shutil


def read_file_lines(file_path):
    with open(file_path, "r") as f:
        return [line.rstrip() for line in f]


def copy_images_and_create_list(src_image_paths, dest_dir, equations_file, equations):
    os.mkdir(dest_dir)
    for i, src_img in enumerate(src_image_paths):
        shutil.copy(src_img, os.path.join(dest_dir, f"{i}.png"))

    with open(equations_file, "w") as f:
        f.write("\n".join(equations))


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "dataset/data")
images_dir = os.path.join(data_dir, "formula_images_processed")
formulas_file = os.path.join(data_dir, "im2latex_formulas.final.lst")
train_file = os.path.join(data_dir, "im2latex_train_filter.lst")
val_file = os.path.join(data_dir, "im2latex_validate_filter.lst")

formulas = read_file_lines(formulas_file)
train_info = read_file_lines(train_file)
val_info = read_file_lines(val_file)

train_formula_line_nos = [int(line.split()[1]) for line in train_info]
train_formula_images = [
    os.path.join(images_dir, line.split()[0]) for line in train_info
]
train_formulas = [formulas[line_no] for line_no in train_formula_line_nos]

val_formula_line_nos = [int(line.split()[1]) for line in val_info]
val_formula_images = [os.path.join(images_dir, line.split()[0]) for line in val_info]
val_formulas = [formulas[line_no] for line_no in val_formula_line_nos]

# Remove train and val directories if they exist
print("Removing old train and val directories...")
shutil.rmtree(os.path.join(data_dir, "train"), ignore_errors=True)
shutil.rmtree(os.path.join(data_dir, "val"), ignore_errors=True)

print("Creating train and val directories...")
copy_images_and_create_list(
    train_formula_images,
    os.path.join(data_dir, "train"),
    os.path.join(data_dir, "train.lst"),
    train_formulas,
)
copy_images_and_create_list(
    val_formula_images,
    os.path.join(data_dir, "val"),
    os.path.join(data_dir, "val.lst"),
    val_formulas,
)

print("Done!")
