#Видалення пошкоджених зображень
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def clean_images(dir_path: Path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    removed=0
    for path in dir_path.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_extensions:
            continue
        try:
            with Image.open(str(path)) as img:
                img.verify()
        except (OSError, ValueError, UnidentifiedImageError) as e:
            path.unlink(missing_ok=True)
            removed += 1
    print(f"Remove {removed} images")
