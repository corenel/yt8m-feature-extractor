"""Load decoded frame dataset from folders.

Partially forked from torchvision.datasets.folder
"""

import os

import torch.utils.data as data
from PIL import Image
from torchvision import get_image_backend

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class FrameImage(data.Dataset):
    """Dataset for video frames from folders."""

    def __init__(self, root, transform=None):
        """Init FrameImage dataset."""
        super(FrameImage, self).__init__()
        images = self.list_image_files(root)
        if len(images) == 0:
            raise(RuntimeError(
                "Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.transform = transform
        self.images = images
        self.loader = self.default_loader

    def __getitem__(self, index):
        """Get frames from video."""
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """Get number of the frames."""
        return len(self.images)

    def is_image_file(self, filename):
        """Check if one file is an image."""
        return any(filename.endswith(extension)
                   for extension in IMG_EXTENSIONS)

    def list_image_files(self, data_dir):
        """List all image files."""
        images = []
        data_dir = os.path.expanduser(data_dir)
        for root, sub, filenames, in sorted(os.walk(data_dir)):
            for filename in filenames:
                if self.is_image_file(filename):
                    path = os.path.join(root, filename)
                    images.append(path)
        return images

    def pil_loader(self, path):
        """PIL image loader."""
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def accimage_loader(self, path):
        """Accimage image loader."""
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)

    def default_loader(self, path):
        """Get default image loader for torchvision."""
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)
