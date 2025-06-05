import os

npy_dir = './data/knife_preprocessed'
split_list_path = os.path.join(npy_dir, 'train_split.lst')

with open(split_list_path, 'w') as f:
    for file in os.listdir(npy_dir):
        if file.endswith('.npy'):
            f.write(file + '\n')

print(f"✅ Fixed: {split_list_path}")
