import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

chrome_formulas_file = os.path.join(project_dir, 'dataset/data/CROHME_math.txt')
train_equations_file = os.path.join(project_dir, 'dataset/data/pdfmath.txt')

with open(chrome_formulas_file, 'r') as f:
    chrome_formulae = [line.rstrip() for line in f.readlines()]

with open(train_equations_file, 'r') as f:
    train_equations = [line.rstrip() for line in f.readlines()]

print("Number of formulae in chrome: ", len(chrome_formulae))
print("Number of formulae in train: ", len(train_equations))
images_dir = os.listdir(os.path.join(project_dir, 'dataset/data/images'))
train_images_dir = os.listdir(os.path.join(project_dir, 'dataset/data/train'))

print("Building train formulae list...")
train_formulae_list = []
for image_name in images_dir:
    line_no = int(image_name.split('.')[0])
    train_formulae_list.append(chrome_formulae[line_no])

for image_name in train_images_dir:
    try:
        line_no = int(image_name.split('.')[0])
        train_formulae_list.append(train_equations[line_no])
    except:
        print("Error: ", os.path.join(project_dir, 'dataset/data/train', image_name))

print("Number of formulae in train_formulae_list: ", len(train_formulae_list))
print("Writing to all_train_equations.txt...")
# write to a file called all_equations.txt
with open(os.path.join(project_dir, 'dataset/data/all_train_equations.txt'), 'w') as f:
    f.write('\n'.join(train_formulae_list))
    f.write('\n')

print("Done writing to all_train_equations.txt")