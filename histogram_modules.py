import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
#           FILE SELECTION HELPER
# ==========================================

def select_image(prompt_text="Select an image"):
    """
    Tries to list images from a local 'Data' folder. 
    If not found, asks user for manual path input.
    """
    # Determine the directory where this script is located
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd() # Fallback for some interactive environments
        
    data_dir = os.path.join(base_dir, 'Data')
    
    found_files = []
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        found_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)])
        
    # If no Data folder or no images, direct manual input
    if not found_files:
        path = input(f"{prompt_text} (enter full path): ").strip().strip('"')
        return path
        
    # Display available files
    print(f"\n--- {prompt_text} ---")
    print(f"Looking in: {data_dir}")
    for i, fname in enumerate(found_files):
        print(f"  [{i+1}] {fname}")
    print("  [0] Enter a different path manually")
    
    choice = input("Enter your choice (number or path): ").strip().strip('"')
    
    # Handle selection
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(found_files):
            return os.path.join(data_dir, found_files[idx-1])
        elif idx == 0:
            return input("Enter full image path: ").strip().strip('"')
    
    # If user entered a string/path directly
    return choice

# ==========================================
#              CORE UTILITIES
# ==========================================

def imread_gray(path):
    """Read an image as uint8 grayscale, raise error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}. Check file format.")
    return img

def _norm01(x):
    """Normalize array to [0,1]."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)

# ==========================================
#       HISTOGRAM EQUALIZATION MODULE
# ==========================================

def histogram(img: np.ndarray, bins: int = 256):
    hist = np.bincount(img.ravel(), minlength=bins).astype(np.int64)[:bins]
    return hist

def cdf(hist: np.ndarray):
    c = np.cumsum(hist)
    total = c[-1]
    return np.zeros_like(c, dtype=np.float64) if total == 0 else c / total

def equalize(img: np.ndarray, bins: int = 256):
    hist = histogram(img, bins=bins)
    cdf_vals = cdf(hist)
    T = np.round(255 * cdf_vals).astype(np.uint8)
    return T[img], T

def adaptive_equalize(gray: np.ndarray, tile_grid=(8, 8), bins: int = 256) -> np.ndarray:
    """Adaptive Histogram Equalization (AHE)."""
    gray = gray.astype(np.uint8, copy=False)
    H, W = gray.shape
    n_ty, n_tx = tile_grid
    y_edges = np.linspace(0, H, n_ty + 1, dtype=int)
    x_edges = np.linspace(0, W, n_tx + 1, dtype=int)

    # Compute LUT per tile
    T_tiles = np.zeros((n_ty, n_tx, bins), dtype=np.uint8)
    for iy in range(n_ty):
        y0, y1 = y_edges[iy], y_edges[iy + 1]
        for ix in range(n_tx):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            tile = gray[y0:y1, x0:x1]
            hist = histogram(tile, bins=bins)
            cdf_vals = cdf(hist)
            T_tiles[iy, ix, :] = np.round(255 * cdf_vals).astype(np.uint8)

    # Interpolate
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    def get_weights(pos, centers):
        n = centers.shape[0]
        if pos <= centers[0]: return 0, 0, 1.0, 0.0
        if pos >= centers[-1]: return n - 1, n - 1, 1.0, 0.0
        idx = np.searchsorted(centers, pos)
        i0, i1 = idx - 1, idx
        t = (pos - centers[i0]) / (centers[i1] - centers[i0])
        return i0, i1, 1.0 - t, t

    eq = np.empty_like(gray, dtype=np.uint8)
    # Optimized loop using indices
    for y in range(H):
        iy0, iy1, wy0, wy1 = get_weights(y, y_centers)
        for x in range(W):
            ix0, ix1, wx0, wx1 = get_weights(x, x_centers)
            v = int(gray[y, x])
            v_tl = int(T_tiles[iy0, ix0, v])
            v_tr = int(T_tiles[iy0, ix1, v])
            v_bl = int(T_tiles[iy1, ix0, v])
            v_br = int(T_tiles[iy1, ix1, v])
            top = wx0 * v_tl + wx1 * v_tr
            bottom = wx0 * v_bl + wx1 * v_br
            eq[y, x] = np.uint8(np.clip(round(wy0 * top + wy1 * bottom), 0, 255))
    return eq

# ==========================================
#        TEMPLATE MATCHING MODULE
# ==========================================

def compute_normalized_histogram_1d(values: np.ndarray, bins: int = 20) -> np.ndarray:
    hist = np.bincount(values.ravel(), minlength=bins).astype(np.int64)[:bins]
    h = hist.astype(np.float64)
    total = h.sum()
    return h / total if total > 0 else h

def earth_movers_distance_1d(hist1, hist2):
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    return float(np.abs(cdf1 - cdf2).sum())

def autocrop_dark_padding(tpl_gray, thresh=5):
    m = tpl_gray > thresh
    if not m.any(): return tpl_gray
    ys, xs = np.where(m)
    return tpl_gray[ys.min():ys.max()+1, xs.min():xs.max()+1]

def hist_emd_score_map(scene, template, bins=20, dark_thresh=5):
    h_tpl, w_tpl = template.shape
    h_sc, w_sc = scene.shape
    
    mask_tpl = template > dark_thresh
    hist_tpl = compute_normalized_histogram_1d(template[mask_tpl], bins=bins)
    
    score_h, score_w = h_sc - h_tpl + 1, w_sc - w_tpl + 1
    score_map = np.zeros((score_h, score_w), dtype=np.float64)
    
    print(f"  > Computing EMD Map ({score_h}x{score_w})...")
    for r in range(score_h):
        for c in range(score_w):
            window = scene[r:r+h_tpl, c:c+w_tpl]
            # Use template mask on window logic (simplification of notebook logic)
            window_vals = window[mask_tpl]
            hist_win = compute_normalized_histogram_1d(window_vals, bins=bins)
            score_map[r, c] = -earth_movers_distance_1d(hist_tpl, hist_win)
            
    return score_map

def best_box_from_score(score_map, tpl_shape):
    idx = np.argmax(score_map)
    r, c = np.unravel_index(idx, score_map.shape)
    h, w = tpl_shape
    return (c, r, c+w, r+h), score_map[r, c]

# --- MTM SLT ---

def make_bin_edges(k: int):
    return np.linspace(0, 256, k + 1, dtype=np.int32)

def slice_indices(img, edges):
    return np.digitize(img.ravel(), edges[1:-1], right=False).reshape(img.shape) - 1

def mtm_slt_score_map(image, template, k=10, alpha=0.3):
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    h, w = template.shape
    m = h * w
    
    edges = make_bin_edges(k)
    ones = np.ones((h, w), dtype=np.float32)
    
    W1 = cv.matchTemplate(image, ones, cv.TM_CCORR)
    W2 = cv.matchTemplate(image**2, ones, cv.TM_CCORR)
    D2 = np.maximum(W2 - (W1**2)/m, 1e-10)
    
    D1 = np.zeros_like(W1, dtype=np.float32)
    tpl_idx = slice_indices(template.astype(np.uint8), edges)
    
    for j in range(k):
        p_j = (tpl_idx == j).astype(np.float32)
        n_j = np.sum(p_j)
        if n_j < 1e-10: continue
        
        T_corr = cv.matchTemplate(image, p_j, cv.TM_CCORR)
        D1 += (T_corr**2) / (n_j + alpha)
        
    D = (W2 - D1) / D2
    return 1.0 - np.clip(D, 0.0, 1.0)

# ==========================================
#           VARIANTS (ROBUSTNESS)
# ==========================================

def brightness_shift(img, shift=50):
    return np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

def contrast_stretch(img, alpha=1.5):
    mean = np.mean(img)
    return np.clip((img.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)

# ==========================================
#               MAIN TASKS
# ==========================================

def get_mtm_slt_params():
    """Gets and validates MTM-SLT parameters from the user."""
    while True:
        try:
            k_str = input("Enter number of bins (k) for MTM-SLT [2-256] (e.g., 10): ").strip()
            k = int(k_str)
            if not (2 <= k <= 256):
                raise ValueError("Number of bins (k) must be between 2 and 256.")

            alpha_str = input("Enter smoothing parameter (alpha) for MTM-SLT [>0-10.0] (e.g., 0.3): ").strip()
            alpha = float(alpha_str)
            if not (0.0 < alpha <= 10.0):
                raise ValueError("Smoothing parameter (alpha) must be a positive value, up to 10.0.")

            return k, alpha
        except ValueError as e:
            print(f"Error: {e}")
        except Exception:
            print("Invalid input. Please enter valid numbers.")


def get_grid_size():
    """Gets and validates the grid size for AHE from the user."""
    while True:
        try:
            grid_str = input("Enter grid size for Adaptive HE (e.g., '8,8'): ").strip()
            parts = [int(p.strip()) for p in grid_str.split(',')]
            if len(parts) != 2:
                raise ValueError("Please enter two numbers separated by a comma.")
            rows, cols = parts
            if not (1 <= rows <= 20 and 1 <= cols <= 20):
                raise ValueError("Both grid dimensions must be between 1 and 20.")
            return (rows, cols)
        except ValueError as e:
            print(f"Error: {e}")
        except Exception:
            print("Invalid input. Please enter in the format 'rows,cols', e.g., '8,8'.")


def run_equalization_task():
    print("\n--- Global vs Adaptive Histogram Equalization ---")
    path = select_image("Select Source Image")
    
    try:
        src = imread_gray(path)
        
        print("Processing Global HE...")
        eq_global, _ = equalize(src)
        
        # Get grid size for adaptive equalization
        grid_size = get_grid_size()
        
        print(f"Processing Adaptive HE with grid {grid_size}...")
        eq_adapt = adaptive_equalize(src, tile_grid=grid_size)
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(src, cmap='gray'); axes[0].set_title(f"Original\n{os.path.basename(path)}")
        axes[1].imshow(eq_global, cmap='gray'); axes[1].set_title("Global HE")
        axes[2].imshow(eq_adapt, cmap='gray'); axes[2].set_title(f"Adaptive HE {grid_size}")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

def run_template_matching_task():
    print("\n--- Template Matching (Hist EMD vs MTM SLT) ---")
    s_path = select_image("Select SCENE Image")
    t_path = select_image("Select TEMPLATE Image")
    
    try:
        scene = imread_gray(s_path)
        tpl = imread_gray(t_path)
        
        print("1. Running Histogram EMD (Slow)...")
        tpl_c = autocrop_dark_padding(tpl)
        S_emd = hist_emd_score_map(scene, tpl_c, bins=20)
        (x0, y0, x1, y1), score_emd = best_box_from_score(S_emd, tpl_c.shape)
        
        # Get MTM-SLT parameters from user
        k, alpha = get_mtm_slt_params()

        print(f"2. Running MTM SLT (Fast) with k={k}, alpha={alpha}...")
        S_mtm = mtm_slt_score_map(scene, tpl, k=k, alpha=alpha)
        (xm0, ym0, xm1, ym1), score_mtm = best_box_from_score(S_mtm, tpl.shape)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # EMD
        axes[0,0].imshow(scene, cmap='gray')
        axes[0,0].add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, ec='orange', fc='none', lw=2))
        axes[0,0].set_title(f"Hist EMD Match (Score: {score_emd:.2f})")
        axes[0,1].imshow(_norm01(S_emd), cmap='jet')
        axes[0,1].set_title("EMD Score Map")
        
        # MTM
        axes[1,0].imshow(scene, cmap='gray')
        axes[1,0].add_patch(patches.Rectangle((xm0, ym0), xm1-xm0, ym1-ym0, ec='lime', fc='none', lw=2))
        axes[1,0].set_title(f"MTM SLT Match (k={k}, α={alpha}, Score: {score_mtm:.2f})")
        axes[1,1].imshow(_norm01(S_mtm), cmap='jet')
        axes[1,1].set_title("MTM Score Map")
        
        for ax in axes.flat: ax.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

def run_robustness_test():
    print("\n--- Robustness Test for MTM-SLT ---")
    s_path = select_image("Select SCENE Image")
    t_path = select_image("Select TEMPLATE Image")
    
    try:
        scene = imread_gray(s_path)
        tpl = imread_gray(t_path)
        
        # Get MTM-SLT parameters from user
        k, alpha = get_mtm_slt_params()

        variants = {
            "Original": scene,
            "Bright (+50)": brightness_shift(scene, 50),
            "Contrast (x1.5)": contrast_stretch(scene, 1.5)
        }
        
        fig, axes = plt.subplots(len(variants), 3, figsize=(12, 4*len(variants)))
        fig.suptitle(f'MTM-SLT Robustness (k={k}, α={alpha})', fontsize=16)

        for i, (name, img_var) in enumerate(variants.items()):
            print(f"Processing variant: {name}...")
            S = mtm_slt_score_map(img_var, tpl, k=k, alpha=alpha)
            (x, y, x2, y2), score = best_box_from_score(S, tpl.shape)
            
            current_axes = axes[i] if len(variants) > 1 else axes
            
            current_axes[0].imshow(img_var, cmap='gray')
            current_axes[0].set_title(name)
            
            current_axes[1].imshow(img_var, cmap='gray')
            current_axes[1].add_patch(patches.Rectangle((x, y), x2-x, y2-y, ec='lime', fc='none', lw=2))
            current_axes[1].set_title(f"Detected (Score: {score:.2f})")
            
            current_axes[2].imshow(_norm01(S), cmap='jet')
            current_axes[2].set_title("Score Map")
            
            for ax in current_axes: ax.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

def main():
    while True:
        print("\n==========================================")
        print(" Integrated Image Processing Tool v2")
        print("==========================================")
        print("1. Histogram Equalization (Global & Adaptive)")
        print("2. Template Matching (EMD vs MTM)")
        print("3. Robustness Test for the SLT+MTM")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            run_equalization_task()
        elif choice == '2':
            run_template_matching_task()
        elif choice == '3':
            run_robustness_test()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()