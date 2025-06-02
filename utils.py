import numpy as np
import cv2
import matplotlib.pyplot as plt
import typing as tp

def convolute_image(im: np.ndarray, a: int, b: int, kernel: tp.Callable, method_of_ext: tp.Literal['reflect', 'edge', 'symmetric', 'wrap']) -> np.ndarray:
    """Совершить свертку по данному фильтру и радиусу

    Args:
        im (np.ndarray): исходное изображение
        a (int): высота полуфильтра
        b (int): длина полуфильтра
        kernel (tp.Callable): функция фильтра
        method_of_ext (tp.Literal[reflect, edge, symmetric, wrap]): метод расширени изображения для обработки границ

    Returns:
        np.ndarray: обработанная картинка
    """
    im_ext = np.pad(im, ((a, a), (b, b)), mode=method_of_ext)
    kernel_matrix = kernel(a, b)
    res = np.zeros_like(im)

    row, col = im.shape[0], im.shape[1]

    for I in range(row):
        for J in range(col):
            for i in range(-a, a + 1):
                for j in range(-b, b + 1):
                    res[I, J] += kernel_matrix[i + a, j + b] * im_ext[I + a - i, J + b - j]
    
    return res