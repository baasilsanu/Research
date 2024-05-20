from PIL import Image
import os


def batch_crop_images(src_folder, dst_folder, crop_area):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Iterate over all image files in the source folder
    i = 0
    for image_name in os.listdir(src_folder):
        i += 1
        print(f"Doing image {i}")
        if image_name.lower().endswith(('.png')):  # Check for image files
            image_path = os.path.join(src_folder, image_name)
            img = Image.open(image_path)

            # Crop the image
            cropped_img = img.crop(crop_area)

            # Save the cropped image to the destination folder
            cropped_img_path = os.path.join(dst_folder, image_name)
            cropped_img.save(cropped_img_path)


# Define the source and destination folders
src_folder = './generatedImages'
dst_folder = './imagesForAnalysis'

# Define the crop area (left, upper, right, lower)
crop_area = (0, 0, 903, 800)  

batch_crop_images(src_folder, dst_folder, crop_area)
