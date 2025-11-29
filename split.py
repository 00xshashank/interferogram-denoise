import os
import random
import shutil

all_images = os.listdir('data/image')
print(f"Number of training images: {len(all_images)}")

all_labels = os.listdir('data/target')
print(f"Number of test images: {len(all_labels)}")

os.makedirs("data\\train", exist_ok=True)
os.makedirs("data\\val", exist_ok=True)
os.makedirs("data\\test", exist_ok=True)

total_imgs_count = len(all_images)
train_imgs_count = int(total_imgs_count * 0.7)
val_imgs_count = int(total_imgs_count * 0.15)
test_imgs_count = int(val_imgs_count)

print(f"Size of: train: {train_imgs_count}, val: {val_imgs_count}, test: {test_imgs_count}")

random.shuffle(all_images)
train_imgs = all_images[:train_imgs_count]
val_imgs = all_images[train_imgs_count+1:train_imgs_count+val_imgs_count]
test_imgs = all_images[train_imgs_count+val_imgs_count+1:]

print(f"Length of: train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")

os.makedirs('data\\train\\images', exist_ok=True)
os.makedirs('data\\train\\labels', exist_ok=True)
for img in train_imgs:
    src_path = os.path.join('data\\image', img)
    dest_path = os.path.join('data\\train\\images', img)
    shutil.copy(src_path, dest_path)

    src_label_path = os.path.join('data\\target', img.replace('image', 'target'))
    dest_label_path = os.path.join('data\\train\\labels', img.replace('image', 'target'))
    shutil.copy(src_label_path, dest_label_path)


os.makedirs('data\\val\\images', exist_ok=True)
os.makedirs('data\\val\\labels', exist_ok=True)
for img in val_imgs:
    src_path = os.path.join('data\\image', img)
    dest_path = os.path.join('data\\val\\images', img)
    shutil.copy(src_path, dest_path)

    src_label_path = os.path.join('data\\target', img.replace('image', 'target'))
    dest_label_path = os.path.join('data\\val\\labels', img.replace('image', 'target'))
    shutil.copy(src_label_path, dest_label_path)


os.makedirs('data\\test\\images', exist_ok=True)
os.makedirs('data\\test\\labels', exist_ok=True)
for img in test_imgs:
    src_path = os.path.join('data\\image', img)
    dest_path = os.path.join('data\\test\\images', img)
    shutil.copy(src_path, dest_path)

    src_label_path = os.path.join('data\\target', img.replace('image', 'target'))
    dest_label_path = os.path.join('data\\test\\labels', img.replace('image', 'target'))
    shutil.copy(src_label_path, dest_label_path)