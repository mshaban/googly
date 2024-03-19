import os


def find_images(directory_path):
    """
    Finds all images in the given directory and its subdirectories, returning their paths.

    :param directory_path: The path to the directory to search for images.
    :return: A list of paths to image files, relative to the input directory path.
    """
    # Define a list of common image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp"}
    print(directory_path)
    # Initialize a list to hold the paths of found image files
    image_paths = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file has one of the image extensions
            if os.path.splitext(file)[1].lower() in image_extensions:
                # Construct the full path to the file
                full_path = os.path.join(root, file)
                # Make the path relative to the input directory and add it to the list
                relative_path = os.path.relpath(full_path, directory_path)
                image_paths.append(relative_path)

    return image_paths
