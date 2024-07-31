from pathlib import Path
from typing import TypeAlias
import sys
import os

import cv2

number : TypeAlias = int | float

UM_PER_PX = 2 * 3.45 / 4  # subsampling * µm/px / optical zoom

def segment_image(
        source_path,
        target_size: (int, int) = (1224, 1024),
        overlap: (number, number) = (500/UM_PER_PX, 500/UM_PER_PX)
    ):
    source_image = cv2.imread(source_path)
    path = Path(source_path)
    recut_directory = path.parent.joinpath(path.stem + "_recut_images")
    os.makedirs(recut_directory)
    source_size_px = source_image.shape[:2]
    x_bounds = segment_axis(source_size_px[1], target_size[0], overlap[0])
    y_bounds = segment_axis(source_size_px[0], target_size[1], overlap[1])
    for x_min, x_max in x_bounds:
        for y_min, y_max in y_bounds:
            image_path = recut_directory.joinpath(f"X{x_min}_Y{y_min}.png")
            print("\r"+image_path.name, end="")
            image = source_image[y_min:y_max, x_min:x_max, :]
            cv2.imwrite(str(image_path), image)


def segment_axis(source: int, target: int, overlap: int | float):
    # get the best number of segments to fit the target size
    overlap = int(overlap)
    segment_count = round((source - overlap)/(target - overlap))
    segment_size = overlap + (source - overlap)//segment_count
    step = segment_size - overlap
    remainder = source - overlap - step * segment_count
    ranges = [[step * i, step * i + segment_size] for i in range(segment_count)]
    ranges[-1][1] = source
    print({"count": segment_count, "size": segment_size, "remainder": remainder})
    return ranges

if __name__ == "__main__":
    im_path = sys.argv[1]
    if not Path(im_path).exists():
        raise FileNotFoundError("source image does not exist")
    segment_image(im_path)
