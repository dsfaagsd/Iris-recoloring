from bisect import bisect_left, bisect_right
from os import PathLike
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
import sys
from typing import Sequence, Tuple, Union
from fdlite.face_detection import FaceDetection, FaceDetectionModel
from fdlite.face_landmark import FaceLandmark, face_detection_to_roi
from fdlite.iris_landmark import IrisLandmark, IrisResults
from fdlite.iris_landmark import iris_roi_from_face_landmarks
from fdlite.transform import bbox_from_landmarks

_Point = Tuple[int, int]
_Size = Tuple[int, int]
_Rect = Tuple[int, int, int, int]

COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "violet": (238, 130, 238),
    "indigo": (75, 0, 130),
    "brown": (165, 42, 42),
    "tan": (210, 180, 140),
    "crimson": (220, 20, 60),
    "lime": (50, 205, 50),
    "olive": (128, 128, 0),
    "navy": (0, 0, 128),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "teal": (0, 128, 128),
    "maroon": (128, 0, 0),
    "peach": (255, 218, 185),
    "beige": (245, 245, 220),
    "lavender": (230, 230, 250),
    "coral": (255, 127, 80),
    "salmon": (250, 128, 114),
    "khaki": (240, 230, 140),
    "mint": (189, 252, 201),
    "skyblue": (135, 206, 235),
    "chartreuse": (127, 255, 0),
    "peru": (205, 133, 63),
    "plum": (221, 160, 221),
    "orchid": (218, 112, 214),
    "slateblue": (106, 90, 205),
    "wheat": (245, 222, 179),
    "goldenrod": (218, 165, 32),
    "midnightblue": (25, 25, 112),
    "turquoise": (64, 224, 208),
    "azure": (240, 255, 255),
    "ivory": (255, 255, 240),
    "snow": (255, 250, 250),
    "honeydew": (240, 255, 240),
    "moccasin": (255, 228, 181),
    "rosybrown": (188, 143, 143),
    "sienna": (160, 82, 45),
    "seagreen": (46, 139, 87),
    "steelblue": (70, 130, 180),
    "lightsteelblue": (176, 196, 222),
}


def recolor_iris(
    image: PILImage,
    iris_results: IrisResults,
    iris_color: Tuple[int, int, int]
) -> PILImage:
    iris_location, iris_size = _get_iris_location(iris_results, image.size)
    eye_image = image.transform(iris_size, Image.EXTENT, data=iris_location)
    eye_image = eye_image.convert(mode='L')
    eye_image = ImageOps.colorize(eye_image, 'black', 'white', mid=iris_color)
    mask = _get_iris_mask(iris_results, iris_location, iris_size, image.size)
    image.paste(eye_image, iris_location, mask)
    return image

def _get_iris_location(
    results: IrisResults, image_size: _Size
) -> Tuple[_Rect, _Size]:
    bbox = bbox_from_landmarks(results.iris).absolute(image_size)
    width, height = int(bbox.width + 1), int(bbox.height + 1)
    size = (width, height)
    left, top = int(bbox.xmin), int(bbox.ymin)
    location = (left, top, left + width, top + height)
    return location, size

def _get_iris_mask(
    results: IrisResults,
    iris_location: _Rect,
    iris_size: _Size,
    image_size: _Size
) -> PILImage:
    left, top, _, bottom = iris_location
    iris_width, iris_height = iris_size
    img_width, img_height = image_size
    eyeball_sorted = sorted([(int(pt.x * img_width), int(pt.y * img_height))
                             for pt in results.eyeball_contour])
    bbox = bbox_from_landmarks(results.eyeball_contour).absolute(image_size)
    x_ofs = left
    y_ofs = top
    y_start = int(max(bbox.ymin, top))
    y_end = int(min(bbox.ymax, bottom))
    mask = np.zeros((iris_height, iris_width), dtype=np.uint8)
    a = iris_width // 2
    b = iris_height // 2
    cx = left + a
    cy = top + b
    box_center_y = int(bbox.ymin + bbox.ymax) // 2
    b_sqr = b**2
    for y in range(y_start, y_end):
        x = int(a * np.math.sqrt(b_sqr - (y-cy)**2) / b)
        x0, x1 = cx - x, cx + x
        A, B = _find_contour_segment(eyeball_sorted, (x0, y))
        left_inside = _is_below_segment(A, B, (x0, y), box_center_y)
        C, D = _find_contour_segment(eyeball_sorted, (x1, y))
        right_inside = _is_below_segment(C, D, (x1, y), box_center_y)
        if not (left_inside or right_inside):
            continue
        elif not left_inside:
            x0 = int(max((B[0] - A[0])/(B[1] - A[1]) * (y - A[1]) + A[0], x0))
        elif not right_inside:
            x1 = int(min((D[0] - C[0])/(D[1] - C[1]) * (y - C[1]) + C[0], x1))
        mask[(y - y_ofs), int(x0 - x_ofs):int(x1 - x_ofs)] = 255
    return Image.fromarray(mask, mode='L')

def _is_below_segment(A: _Point, B: _Point, C: _Point, mid: int) -> bool:
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    if not dx:
        return A[1] <= C[1] and B[1] >= C[1]
    x = (C[0] - A[0]) / dx
    m = dy / dx
    y = x * m + A[1]
    sign = -1 if A[1] > mid else 1
    return sign * (C[1] - y) > 0

def _find_contour_segment(
    contour: Sequence[_Point], point: _Point
) -> Tuple[_Point, _Point]:
    def distance(a: _Point, b: _Point) -> int:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    MAX_IDX = len(contour)-1
    left_idx = max(bisect_left(contour, point) - 1, 0)
    right_idx = min(bisect_right(contour, point), MAX_IDX)
    d = distance(point, contour[left_idx])
    while left_idx > 0 and d > distance(point, contour[left_idx - 1]):
        left_idx -= 1
        d = distance(point, contour[left_idx - 1])
    d = distance(point, contour[right_idx])
    while right_idx < MAX_IDX and d > distance(point, contour[right_idx + 1]):
        right_idx += 1
        d = distance(point, contour[right_idx + 1])
    return (contour[left_idx], contour[right_idx])

def main(image_file: Union[str, PathLike]) -> None:
    print("Available colors:", ", ".join(COLOR_MAP.keys()))
    color_input = input("Enter your desired eye color: ").strip().lower()
    iris_color = COLOR_MAP.get(color_input, (0, 0, 0))

    img = Image.open(image_file)
    face_detection = FaceDetection(FaceDetectionModel.BACK_CAMERA)
    detections = face_detection(img)
    if not len(detections):
        print('No face detected :(')
        exit(0)
    face_roi = face_detection_to_roi(detections[0], img.size)

    face_landmarks = FaceLandmark()
    landmarks = face_landmarks(img, face_roi)
    eyes_roi = iris_roi_from_face_landmarks(landmarks, img.size)

    iris_landmarks = IrisLandmark()
    left_eye_roi, right_eye_roi = eyes_roi
    left_eye_results = iris_landmarks(img, left_eye_roi)
    right_eye_results = iris_landmarks(img, right_eye_roi, is_right_eye=True)

    recolor_iris(img, left_eye_results, iris_color=iris_color)
    recolor_iris(img, right_eye_results, iris_color=iris_color)

    img.show()

if __name__ == '__main__':
    image_path = r"       " #enter your image path
    main(image_path)
