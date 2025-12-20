import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os

# ==========================================
#        PART 1: HELPER FUNCTIONS
# ==========================================

def show_image(im, title=None, cmap="gray", vmin=0, vmax=255):
    """Simple visualization helper for grayscale images."""
    im_disp = np.asarray(im, dtype=np.float32)
    if vmin is not None or vmax is not None:
        im_disp = np.clip(im_disp, vmin, vmax)
    plt.figure()
    plt.imshow(im_disp, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

def psnr(clean_im: np.ndarray, test_im: np.ndarray, max_value: float = 255.0):
    """Calculate PSNR between two images."""
    clean = clean_im.astype(np.float32)
    test = test_im.astype(np.float32)
    mse = np.mean((clean - test) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((max_value ** 2) / mse)
    return psnr_value

def load_image_from_path(path):
    """Loads an image and converts it to grayscale 0-255 uint8."""
    try:
        img = plt.imread(path)
        # Convert to grayscale if it's RGB/RGBA
        if img.ndim == 3:
            if img.shape[2] == 4: # RGBA
                img = img[:, :, :3]
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Ensure range 0-255 if strictly float 0-1
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# ==========================================
#        PART 2: NOISE GENERATORS
# ==========================================

def add_sp_noise(im: np.ndarray, p: float):
    """Add Salt & Pepper noise."""
    noisy_im = im.copy()
    probs = np.random.rand(*im.shape)
    noisy_im[probs < (p / 2)] = 0
    noisy_im[probs > (1 - p / 2)] = 255
    return noisy_im

def add_gaussian_noise(im: np.ndarray, std: float):
    """Add Gaussian noise."""
    noisy_im = im.astype(np.float32).copy()
    noise = np.random.normal(0, std, im.shape)
    noisy_im = noisy_im + noise
    return np.clip(noisy_im, 0, 255).astype(np.uint8)

# ==========================================
#        PART 3: DENOISING FILTERS
# ==========================================

def clean_Gaussian_Noise(im, radius, mask_std):
    """Linear Gaussian Filter (Convolution)."""
    r = int(radius)
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * mask_std**2))
    kernel = kernel / np.sum(kernel)
    denoised_im = convolve2d(im, kernel, mode='same', boundary='symm')
    return denoised_im

def clean_SP_Noise(im, radius):
    """Median Filter for single S&P image."""
    r = int(radius)
    h, w = im.shape
    padded_im = np.pad(im, pad_width=r, mode='edge')
    output_im = np.zeros_like(im)
    
    # Iterate pixels
    for i in range(h):
        for j in range(w):
            window = padded_im[i : i + 2*r + 1, j : j + 2*r + 1]
            output_im[i, j] = np.median(window)
    return output_im

def clean_Gaussian_Noise_Bilateral(im, radius, std_spatial, std_intensity):
    """
    Bilateral Filter implementation.
    Preserves edges by considering both spatial distance and intensity difference.
    """
    r = int(radius)
    h, w = im.shape
    padded_im = np.pad(im, pad_width=r, mode='edge').astype(np.float32)
    output_im = np.zeros_like(im, dtype=np.float32)
    
    # Pre-compute spatial kernel (Gaussian)
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x, y)
    spatial_kernel = np.exp(-(xx**2 + yy**2) / (2 * std_spatial**2))
    
    print(f"Running Bilateral Filter (Radius={r})... This might take a moment.")
    
    for i in range(h):
        for j in range(w):
            # Extract window
            window = padded_im[i : i + 2*r + 1, j : j + 2*r + 1]
            center_val = window[r, r]
            
            # Calculate intensity weights
            intensity_diff = window - center_val
            intensity_kernel = np.exp(-(intensity_diff**2) / (2 * std_intensity**2))
            
            # Combine weights
            weights = spatial_kernel * intensity_kernel
            norm_factor = np.sum(weights)
            
            if norm_factor > 0:
                output_im[i, j] = np.sum(weights * window) / norm_factor
            else:
                output_im[i, j] = center_val
                
    return np.clip(output_im, 0, 255).astype(np.uint8)

def clean_SP_Noise_Multiple(im, p, N):
    """
    Multi-Image Denoising for S&P.
    Generates N noisy copies of 'im' and computes the median to denoise.
    """
    print(f"Generating {N} noisy images and taking the median...")
    h, w = im.shape
    # Stack to hold all images
    stack = np.zeros((N, h, w), dtype=np.float32)

    # Generate N noisy versions
    for k in range(N):
        stack[k, :, :] = add_sp_noise(im, p)

    # Calculate Median across the stack (Z-direction)
    denoised_median = np.median(stack, axis=0)

    # Return the first noisy image (for display purposes) and the final result
    sample_noisy = stack[0, :, :].astype(np.uint8)
    final_result = np.clip(denoised_median, 0, 255).astype(np.uint8)

    return sample_noisy, final_result

# ==========================================
#        PART 4: UI & LOGIC
# ==========================================

def get_image_input():
    """Handles listing files and getting user selection."""
    print(f"Current working directory: {os.getcwd()}")
    data_dir = "Data"
    available_images = []
    
    # Check folder and list files
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = os.listdir(data_dir)
        available_images = [f for f in files if f.lower().endswith(valid_exts)]
    
    selected_path = None
    
    if available_images:
        print(f"Images found in '{data_dir}':")
        for idx, fname in enumerate(available_images):
            print(f"{idx + 1}. {fname}")
        print(f"{len(available_images) + 1}. Enter custom path")
        
        choice = input("Select an image number: ")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_images):
                selected_path = os.path.join(data_dir, available_images[choice_idx])
            else:
                selected_path = input("Enter full path to image: ").strip()
        except ValueError:
            selected_path = input("Enter full path to image: ").strip()
    else:
        print(f"No images found in '{data_dir}'.")
        selected_path = input("Enter full path to image: ").strip()
        
    return selected_path

def main():

    # --- Noise Parameters ---
    P_VAL_SP = 0.3      # Probability for Salt & Pepper noise
    STD_GAUSSIAN = 20.0  # Standard deviation for Gaussian noise

    # --- Denoising Parameters ---
    MEDIAN_RADIUS = 1

    # Gaussian Filter
    GAUSSIAN_RADIUS = 2
    GAUSSIAN_MASK_STD = 1.0

    # Bilateral Filter
    BILATERAL_RADIUS = 2
    BILATERAL_STD_SPATIAL = 20.0
    BILATERAL_STD_INTENSITY = 30.0

    # Multi-Image Median
    MULTI_IMAGE_N = 50       # Number of images in the stack
    # ==========================================

    # Load Image
    image_path = get_image_input()
    
    if not image_path or not os.path.exists(image_path):
        print("Invalid file path.")
        return

    original_img = load_image_from_path(image_path)
    if original_img is None:
        return
    
    print(f"\nLoaded image size: {original_img.shape}")

    # Choose Noising Technique
    print("\nSelect Noising Technique:")
    print("1. Salt & Pepper")
    print("2. Gaussian")
    noise_choice = input("Choice (1/2): ")

    noisy_img = None
    noise_desc = ""

    if noise_choice == '1':
        noisy_img = add_sp_noise(original_img, P_VAL_SP)
        noise_desc = f"S&P Noise (p={P_VAL_SP})"
    elif noise_choice == '2':
        noisy_img = add_gaussian_noise(original_img, STD_GAUSSIAN)
        noise_desc = f"Gaussian Noise (std={STD_GAUSSIAN})"
    else:
        print("Invalid noise choice.")
        return

    # Choose Denoising Technique (decoupled)
    print("\nSelect Denoising Technique:")
    print("1. Median Filter")
    print("2. Gaussian Filter (Linear)")
    print("3. Bilateral Filter")
    print("4. Multi-Image Median (operates on original, not the noisy image above)")
    denoise_choice = input("Choice (1/2/3/4): ")

    denoised_img = None
    filter_desc = ""
    # By default, the noisy image to display is the one we just generated.
    # This will be overridden in the special case of the multi-image denoiser.
    noisy_img_display = noisy_img

    if denoise_choice == '1':
        denoised_img = clean_SP_Noise(noisy_img, MEDIAN_RADIUS)
        filter_desc = f"Median Filter (r={MEDIAN_RADIUS})"

    elif denoise_choice == '2':
        denoised_img = clean_Gaussian_Noise(noisy_img, GAUSSIAN_RADIUS, GAUSSIAN_MASK_STD)
        filter_desc = f"Gaussian Filter (r={GAUSSIAN_RADIUS}, std={GAUSSIAN_MASK_STD})"

    elif denoise_choice == '3':
        denoised_img = clean_Gaussian_Noise_Bilateral(noisy_img, BILATERAL_RADIUS, BILATERAL_STD_SPATIAL, BILATERAL_STD_INTENSITY)
        filter_desc = f"Bilateral (r={BILATERAL_RADIUS}, s_s={BILATERAL_STD_SPATIAL}, s_r={BILATERAL_STD_INTENSITY})"

    elif denoise_choice == '4':
        # This case is special. It ignores the previously generated noisy image
        # and works directly from the original.
        noisy_img_display, denoised_img = clean_SP_Noise_Multiple(original_img, P_VAL_SP, MULTI_IMAGE_N)
        
        # We must override the noise and filter descriptions for the plot.
        noise_desc = f"S&P Noise (p={P_VAL_SP}, 1 of {MULTI_IMAGE_N} shown)"
        filter_desc = f"Multi-Image Median (N={MULTI_IMAGE_N})"

    else:
        print("Invalid denoising choice.")
        return

    # 4. Results and Plotting
    if denoised_img is None:
        print("Denoising was not performed.")
        return

    psnr_noisy = psnr(original_img, noisy_img_display)
    psnr_denoised = psnr(original_img, denoised_img)

    print(f"\n--- Results ---")
    print(f"PSNR (Noisy vs Orig):    {psnr_noisy:.2f} dB")
    print(f"PSNR (Denoised vs Orig): {psnr_denoised:.2f} dB")

    plt.figure(figsize=(16, 6))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")
    plt.axis('off')

    # Noisy
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img_display, cmap='gray', vmin=0, vmax=255)
    plt.title(f"{noise_desc}\nPSNR: {psnr_noisy:.2f} dB")
    plt.axis('off')

    # Denoised
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_img, cmap='gray', vmin=0, vmax=255)
    plt.title(f"{filter_desc}\nPSNR: {psnr_denoised:.2f} dB")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()