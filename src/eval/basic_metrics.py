import numpy as np


def calculate_mse(original_images, processed_images):
    mse_values = []
    for orig, proc in zip(original_images, processed_images):
        # Calculate MSE
        mse = np.mean((orig - proc) ** 2)
        mse_values.append(mse)

    return np.array(mse_values)


def calculate_psnr_from_mse(mse_array, max_pixel_value=255.0):
    """
    Calculate PSNR for each MSE in the given array.

    :param mse_array: Array of MSE values
    :param max_pixel_value: Maximum pixel value (default is 255 for 8-bit images)
    :return: Array of PSNR values
    """
    # Remove any potential NaN or inf values
    valid_mse = mse_array[np.isfinite(mse_array)]
    valid_mse = np.maximum(valid_mse, 1e-10)
    psnr_array = 20 * np.log10(max_pixel_value) - 10 * np.log10(valid_mse)
    return psnr_array
