import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def analyze_image(original_path, upscaled_path):
    # Read images
    original = cv2.imread(original_path)
    upscaled = cv2.imread(upscaled_path)

    # Ensure images are the same size for comparison
    if original.shape != upscaled.shape:
        original = cv2.resize(original, (upscaled.shape[1], upscaled.shape[0]))

    # Convert images to grayscale for some metrics
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist_original = cv2.calcHist([original], [0], None, [256], [0, 256])
    hist_upscaled = cv2.calcHist([upscaled], [0], None, [256], [0, 256])

    # Calculate PSNR
    psnr_value = psnr(original, upscaled)

    # Calculate SSIM
    ssim_value, _ = ssim(original_gray, upscaled_gray, full=True)

    # Calculate edge quality
    edges_original = cv2.Canny(original_gray, 100, 200)
    edges_upscaled = cv2.Canny(upscaled_gray, 100, 200)
    edge_quality = np.sum(edges_upscaled) / np.sum(edges_original)

    # Calculate noise levels
    noise_original = np.std(original_gray)
    noise_upscaled = np.std(upscaled_gray)

    # Calculate texture sharpness
    laplacian_original = cv2.Laplacian(original_gray, cv2.CV_64F)
    laplacian_upscaled = cv2.Laplacian(upscaled_gray, cv2.CV_64F)
    sharpness_original = np.var(laplacian_original)
    sharpness_upscaled = np.var(laplacian_upscaled)

    return {
        'histogram': {
            'original': hist_original.flatten().tolist(),
            'upscaled': hist_upscaled.flatten().tolist()
        },
        'psnr': float(psnr_value),
        'ssim': float(ssim_value),
        'edge_quality': float(edge_quality),
        'noise_level': {
            'original': float(noise_original),
            'upscaled': float(noise_upscaled)
        },
        'texture_sharpness': {
            'original': float(sharpness_original),
            'upscaled': float(sharpness_upscaled)
        }
    }
