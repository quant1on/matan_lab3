import numpy as np
import cv2
import matplotlib.pyplot as plt
import typing as tp

def convolute_image(im: np.ndarray, a: int, b: int, kernel: tp.Callable,
                    method_of_ext: tp.Literal['reflect', 'edge', 'symmetric', 'wrap']) -> np.ndarray:
    """Совершить свертку по данному ядру и радиусу

    Args:
        im (np.ndarray): исходное изображение
        a (int): высота полуфильтра
        b (int): длина полуфильтра
        kernel (tp.Callable): функция фильтра
        method_of_ext (tp.Literal[reflect, edge, symmetric, wrap]): метод расширени изображения для обработки границ

    Returns:
        np.ndarray: обработанная картинка
    """

    if (len(im.shape) == 2):
        im_ext = np.pad(im, ((a, a), (b, b)), mode=method_of_ext).astype(float)
    elif (len(im.shape) == 3):
        im_ext = np.pad(im, ((a, a), (b, b), (0, 0)), mode=method_of_ext).astype(float)
    else:
        raise ValueError("Passed a non-image object!")
    
    res = np.zeros_like(im).astype(float)

    row, col = im.shape[0], im.shape[1]
    kernel_matrix = kernel(a, b)
    for I in range(row):
        for J in range(col):
            for i in range(-a, a + 1):
                for j in range(-b, b + 1):
                    res[I, J] += kernel_matrix[i + a, j + b] * im_ext[I + a - i, J + b - j]
    if (res.max() > 255):
        res = res / res.max() * 255
    return res

def gaussian_kernel(a=1, b=1, sigma=1.0):
    """
    Генерирует гауссово ядро
    
    Параметры:
    a, b: полуразмеры ядра по вертикали и горизонтали
    sigma: параметр размытия
    
    Возвращает:
    Нормализованное гауссово ядро размера (2a+1) x (2b+1)
    """
    size_x, size_y = 2*a+1, 2*b+1
    ax = np.linspace(-a, a, size_x)
    ay = np.linspace(-b, b, size_y)
    x, y = np.meshgrid(ax, ay)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def box_blur_kernel(a=1, b=1):
    """
    Генерирует ядро усредняющего фильтра
    
    Параметры:
    a, b: полуразмеры ядра по вертикали и горизонтали
    
    Возвращает:
    Усредняющее ядро размера (2a+1) x (2b+1)
    """
    size_x, size_y = 2*a+1, 2*b+1
    return np.ones((size_x, size_y)) / (size_x * size_y)

def threshold_filter(image, threshold=128):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.where(gray > threshold, 255, 0).astype(np.uint8)

def median_filter(image, kernel_size=3, mode_of_ext='reflect'):
    kernel_size = abs(kernel_size) + (abs(kernel_size - 1) % 2)
    pad = kernel_size // 2

    if len(image.shape) == 2:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode=mode_of_ext)
    elif len(image.shape) == 3:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode=mode_of_ext)
    else:
        raise ValueError("Passed a non-image object!")
    
    output = np.zeros_like(image)
    
    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(region)
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, :]
                output[i, j, :] = np.median(region, axis=(0, 1))
    return output


def sobel_filter(image: np.ndarray, mode_of_ext: str = 'reflect') -> np.ndarray:
    """
    Применяет фильтр Собеля к изображению для выделения границ.
    
    Args:
        image (np.ndarray): Исходное изображение в градациях серого (2D-массив).
        mode_of_ext (str, optional): Режим расширения при помощи np.pad.
    
    Returns:
        np.ndarray: Изображение, содержащее нормированные значения градиента (0-255).
    """
    pad = 1
    image_ext = np.pad(image, ((pad, pad), (pad, pad)), mode=mode_of_ext)
    rows, cols = image.shape

    gradient_magnitude = np.zeros_like(image, dtype=float)
    
    kernel_x = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]], dtype=float)
    
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=float)
    
    for i in range(rows):
        for j in range(cols):
            window = image_ext[i:i+3, j:j+3]
            
            val_x = np.sum(kernel_x * window)
            val_y = np.sum(kernel_y * window)
            
            gradient_magnitude[i, j] = np.sqrt(val_x**2 + val_y**2)
    
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    return gradient_magnitude.astype(np.uint8)

def compute_fourier_coefficients(image, m=100, n=100):
    """Вычисление коэффициентов Фурье a_jk для изображения.
    Параметры:
    image - np.array(черно-белая картинка)
    m - Количество гармоник по x
    n - Количество гармоник по y
    """
    l1, l2 = image.shape
    x = np.arange(l1)
    y = np.arange(l2)
    j_values = np.arange(-m, m+1)
    k_values = np.arange(-n, n+1)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    a_jk = np.zeros((2*m+1, 2*n+1))
    for j_idx, j in enumerate(j_values):
        for k_idx, k in enumerate(k_values):
            cos_j = np.cos(np.pi * j * X / l1)
            cos_k = np.cos(np.pi * k * Y / l2)
            integrand = image * cos_j * cos_k
            a_jk[j_idx, k_idx] = np.sum(integrand) / (l1 * l2)
    return a_jk

def reconstruct_image(image, new_size, m=100, n=100):
    """Восстановление изображения с новым размером.
    Параметры:
    image - np.array(черно-белая картинка)
    new_size - кортеж чисел с новыми размерами(высота, ширина)
    m - Количество гармоник по x
    n - Количество гармоник по y
    """
    a_jk = compute_fourier_coefficients(image, m, n)
    
    new_l1, new_l2 = new_size
    x_new = np.linspace(0, new_l1-1, new_l1)
    y_new = np.linspace(0, new_l2-1, new_l2)
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
    
    reconstructed = np.zeros((new_l1, new_l2))
    j_values = np.arange(-m, m+1)
    k_values = np.arange(-n, n+1)
    
    for j_idx, j in enumerate(j_values):
        for k_idx, k in enumerate(k_values):
            cos_j = np.cos(np.pi * j * X_new / new_l1)
            cos_k = np.cos(np.pi * k * Y_new / new_l2)
            reconstructed += a_jk[j_idx, k_idx] * cos_j * cos_k
    
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed.astype(np.uint8)


def reconstruct_image_from_coefs(a_jk, new_size, m=100, n=100):
    """Восстановление изображения.
    Параметры:
    a_jk - np.array(матрица коэффицентов посчитанная из оригинальной картинки)
    new_size - пара чисел с размерами(высота, ширина)
    m - Количество гармоник по x
    n - Количество гармоник по y
    """
    # Вычисляем коэфмценты
    
    new_l1, new_l2 = new_size
    x_new = np.linspace(0, new_l1-1, new_l1)
    y_new = np.linspace(0, new_l2-1, new_l2)
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
    
    reconstructed = np.zeros((new_l1, new_l2))
    j_values = np.arange(-m, m+1)
    k_values = np.arange(-n, n+1)
    
    for j_idx, j in enumerate(j_values):
        for k_idx, k in enumerate(k_values):
            cos_j = np.cos(np.pi * j * X_new / new_l1)
            cos_k = np.cos(np.pi * k * Y_new / new_l2)
            reconstructed += a_jk[j_idx, k_idx] * cos_j * cos_k
    
    # Обрезаем значения до диапазона [0, 255] и округляем
    reconstructed = reconstructed/np.max(reconstructed) * 255
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed.astype(np.uint8)


def low_pass_filter(image, m, n, threshold):

    """Обнуляет коэффициенты с индексами выше порога (выделение низких частот).
    Параметры:
    image - np.array(черно-белая картинка)
    new_size - пара чисел с новыми размерами(высота, ширина)
    m - Количество гармоник по x
    n - Количество гармоник по y
    threshold - порог обнуления(расстояние от центра)
    """

    filtered = compute_fourier_coefficients(image, m, n)
    center_j = filtered.shape[0] // 2
    center_k = filtered.shape[1] // 2
    for j in range(filtered.shape[0]):
        for k in range(filtered.shape[1]):
            if abs(j - center_j) > threshold or abs(k - center_k) > threshold:
                filtered[j, k] = 0
    return reconstruct_image_from_coefs(filtered, image.shape, m, n)

def high_pass_filter(image, m, n, threshold):

    """Обнуляет коэффициенты с индексами ниже порога (выделение высоких частот).
    Параметры:
    image - np.array(черно-белая картинка)
    m - Количество гармоник по x
    n - Количество гармоник по y 
    threshold - порог обнуления(расстояние от центра)
    """
    
    filtered = compute_fourier_coefficients(image, m, n)
    center_j = filtered.shape[0] // 2
    center_k = filtered.shape[1] // 2
    for j in range(filtered.shape[0]):
        for k in range(filtered.shape[1]):
            if abs(j - center_j) < threshold and abs(k - center_k) < threshold:
                filtered[j, k] = 0
    return reconstruct_image_from_coefs(filtered, image.shape, m, n)