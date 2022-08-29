from exif import Image
from skimage.transform import rotate
from skimage.util import img_as_ubyte


def auto_rotate(image_rgb, image_path):
    """Rotates image automatically if needed, according to EXIF data.

    Parameters
    ----------
    image_path : str
        Path of the input image.
    image_rgb : 3D array
       RGB image of the lepidopteran, with ruler and tags.

    Returns
    -------
    image_rgb : 3D array
        RGB image, rotated if angle in EXIF data is different than zero.
    """
    angle = read_angle(image_path)

    if angle:
        print(f'Original EXIF image angle: {angle} deg')
    else:
        print(f"Couldn't determine EXIF image angle")

    if angle not in (None, 0):  # angle == 0 does not need untilting
        image_rgb = rotate(image_rgb, angle=angle, resize=True)

    return img_as_ubyte(image_rgb)


def read_angle(image_path):
    """Read angle from image on path, according to EXIF data.

    Parameters
    ----------
    image_path : str
        Path of the input image.

    Returns
    -------
    angle : int or None
        Current orientation of the image in degrees, or None if EXIF data
        cannot be read.
    """
    metadata = Image(image_path)

    try:
        if metadata.has_exif:
            angle = metadata.orientation.value
            # checking possible angles for images.
            angles = {1: 0,  # (top, left)
                      6: 90,  # (right, top)
                      3: 180,  # (bottom, right)
                      8: 270}  # (left, bottom)
            return angles.get(angle, 0)
        else:
            print(f'Cannot evaluate orientation for {image_path}.')
            return None
    except ValueError:  # ... is not a valid TiffByteOrder
        print(f'Cannot evaluate orientation for {image_path}.')
        return None
