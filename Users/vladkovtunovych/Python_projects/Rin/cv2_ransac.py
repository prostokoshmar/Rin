import os
import shutil
import numpy as np
import logging
from skimage import io, filters, color, img_as_float, exposure



import matplotlib.pyplot as plt
from glob import glob
from skimage.filters import median
from sklearn.cluster import KMeans
import cv2
import tempfile
import subprocess


TILE_DIR = None
MATCH_VIS_DIR = "."
matching_ratio = 0.4
matching_angle = 3
min_matches = 4
len_gap = 10
selected_method = "sift"

ransac_enabled = False
ransac_thresh = 5.0
ransac_min_inliers = 4

preprocess_enabled = True

def get_output_paths(method):
    """Повертає імена вихідних файлів і папок для заданого методу."""
    OUTPUT_IMAGE_COMPONENT = f"stitched_result_component_{method}_{{}}.tif"
    OUTPUT_IMAGE_ALL = f"stitched_result_all_{method}.tif"
    MATCH_VIS_ACCEPTED = os.path.join(MATCH_VIS_DIR, f"accepted_{method}")
    MATCH_VIS_REJECTED = os.path.join(MATCH_VIS_DIR, f"rejected_{method}")
    for d in [MATCH_VIS_ACCEPTED, MATCH_VIS_REJECTED]:
        os.makedirs(d, exist_ok=True)
    return OUTPUT_IMAGE_COMPONENT, OUTPUT_IMAGE_ALL, MATCH_VIS_ACCEPTED, MATCH_VIS_REJECTED

def preprocess_image(img, do_preprocess=None):
    """Попередня обробка: denoise + optional CLAHE + contrast stretch + optional unsharp + median.

    If do_preprocess is None, the module-level preprocess_enabled is used. When False, CLAHE
    and unsharp mask are skipped (only gaussian denoise and median filter are applied).
    """
    global preprocess_enabled
    if do_preprocess is None:
        do_preprocess = preprocess_enabled

    img = img_as_float(img)

    
    img = filters.gaussian(img, sigma=0.8)

    if do_preprocess:
        
        try:
            img = exposure.equalize_adapthist(img, clip_limit=0.03)
        except Exception:
            
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))

        
        img = filters.unsharp_mask(img, radius=1.0, amount=1.0)
    else:
        pass

    
    img = median(img)

    return img

def load_tiles(folder):
    
    
    patterns = []
    patterns += glob(os.path.join(folder, "*.tiff"))
    patterns += glob(os.path.join(folder, "*.tif"))
    patterns += glob(os.path.join(folder, "*.png"))
    patterns += glob(os.path.join(folder, "*.gwy"))
    paths = sorted(set(patterns))
    if not paths:
        raise ValueError("No tiles (.tiff, .tif, .png, .gwy) found in the specified directory.")

    
    with tempfile.TemporaryDirectory() as td:
        
        first_path = paths[0]
        if first_path.lower().endswith('.gwy'):
            tmp0 = os.path.join(td, os.path.splitext(os.path.basename(first_path))[0] + ".tiff")
            try:
                subprocess.run(["gwyddion", f"--export={tmp0}", first_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                raise FileNotFoundError("gwyddion is not installed or not found in PATH.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to convert {first_path}: {e.stderr.decode(errors='ignore') if e.stderr else e}")
            sample = io.imread(tmp0)
        else:
            sample = io.imread(first_path)

        if sample.ndim == 3:
            
            if sample.shape[2] == 4:
                try:
                    sample = color.rgba2rgb(sample)
                except Exception:
                    
                    sample = sample[..., :3]
            sample = color.rgb2gray(sample)
        h, w = sample.shape

        tiles, raw_tiles = [], []
        for p in paths:
            try:
                if p.lower().endswith('.gwy'):
                    tmp = os.path.join(td, os.path.splitext(os.path.basename(p))[0] + ".tiff")
                    subprocess.run(["gwyddion", f"--export={tmp}", p], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    img = io.imread(tmp)
                else:
                    img = io.imread(p)
            except FileNotFoundError:
                raise FileNotFoundError("gwyddion is not installed or not found in PATH.")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Conversion failed for {p}: {e.stderr.decode(errors='ignore') if e.stderr else e}")
                continue
            if img.ndim == 3:
                
                if img.shape[2] == 4:
                    try:
                        img = color.rgba2rgb(img)
                    except Exception:
                        img = img[..., :3]
                img = color.rgb2gray(img)
            raw_tiles.append(img_as_float(img))
            tiles.append(preprocess_image(img))

    logging.info(f"Loaded {len(raw_tiles)} tiles of size {h}x{w}")
    return tiles, raw_tiles, h, w, paths

def analyze_match_decision(matches, angle_std, ratio, method="sift"):
   
    explanation = []
    if len(matches) < min_matches:
        explanation.append(f"Відхилено: Недостатньо точок зіставлення ({len(matches)} < {min_matches}).")
    
    if angle_std > np.deg2rad(matching_angle):
        explanation.append(
            f"Відхилено: Стандартне відхилення кутів ({angle_std:.2f} радіан) перевищує поріг ({matching_angle}°).")
    if ratio < matching_ratio:
        explanation.append(
            f"Відхилено: Співвідношення правильних ліній ({ratio:.2%}) нижче порогу ({matching_ratio:.2%}).")
    if not explanation:
        explanation.append(
            f"Прийнято: Достатньо точок ({len(matches)} ≥ {min_matches}), стандартне відхилення кутів ({angle_std:.2f} ≤ {np.deg2rad(matching_angle):.2f} радіан), і співвідношення правильних ліній ({ratio:.2%} ≥ {matching_ratio:.2%}).")
    return "\n".join(explanation)

def match_features_sift(img1, img2, method="sift", nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, ratio_test=0.75):
    
    try:
        a1 = (img1 * 255.0).clip(0, 255).astype(np.uint8) if getattr(img1, 'dtype', None) is not None and img1.dtype.kind == 'f' else img1.astype(np.uint8)
        a2 = (img2 * 255.0).clip(0, 255).astype(np.uint8) if getattr(img2, 'dtype', None) is not None and img2.dtype.kind == 'f' else img2.astype(np.uint8)
    except Exception:
        a1 = img1.astype(np.uint8)
        a2 = img2.astype(np.uint8)

    if a1.ndim == 3:
        if a1.shape[2] == 4:
            try:
                a1 = cv2.cvtColor(a1, cv2.COLOR_RGBA2GRAY)
            except Exception:
                a1 = cv2.cvtColor(a1[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            a1 = cv2.cvtColor(a1, cv2.COLOR_RGB2GRAY)
    if a2.ndim == 3:
        if a2.shape[2] == 4:
            try:
                a2 = cv2.cvtColor(a2, cv2.COLOR_RGBA2GRAY)
            except Exception:
                a2 = cv2.cvtColor(a2[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            a2 = cv2.cvtColor(a2, cv2.COLOR_RGB2GRAY)

    
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
    except Exception:
        
        try:
            sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
        except Exception:
            logging.error("SIFT is not available in your OpenCV build.")
            return None, np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2), dtype=int), 0.0

    kp1, des1 = sift.detectAndCompute(a1, None)
    kp2, des2 = sift.detectAndCompute(a2, None)

    k1 = np.array([[kp.pt[1], kp.pt[0]] for kp in kp1]) if kp1 is not None else np.empty((0, 2))
    k2 = np.array([[kp.pt[1], kp.pt[0]] for kp in kp2]) if kp2 is not None else np.empty((0, 2))

    if des1 is None or des2 is None or len(k1) < min_matches or len(k2) < min_matches:
        logging.info(f"Debug SIFT: Not enough descriptors or keypoints (des1={None if des1 is None else len(des1)}, des2={None if des2 is None else len(des2)}, k1={len(k1)}, k2={len(k2)})")
        return None, k1, k2, np.empty((0, 2), dtype=int), 0.0

    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    try:
        matches_knn = bf.knnMatch(des1, des2, k=2)
    except Exception:
        
        bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        raw = bf2.match(des1, des2)
        raw = sorted(raw, key=lambda x: x.distance)
        matches = np.array([[m.queryIdx, m.trainIdx] for m in raw[:max(min_matches, len(raw))]], dtype=int)
        if len(matches) < min_matches:
            return None, k1, k2, matches, 0.0
        src = k1[matches[:, 0]][:, ::-1]
        dst = k2[matches[:, 1]][:, ::-1]
        disp = dst - src
        angles = np.arctan2(disp[:, 1], disp[:, 0])
        angle_std = float(np.std(angles)) if angles.size > 0 else 0.0
        logging.info(f"Debug SIFT: Angle std = {angle_std:.4f} rad, Matches = {len(matches)}")
        return None, k1, k2, matches, angle_std

    
    good = []
    for pair in matches_knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_test * n.distance:
                good.append((m.queryIdx, m.trainIdx))
        elif len(pair) == 1:
            m = pair[0]
            good.append((m.queryIdx, m.trainIdx))

    if len(good) < min_matches:
        return None, k1, k2, np.array(good, dtype=int), 0.0

    matches = np.array(good, dtype=int)

    
    try:
        src = k1[matches[:, 0]][:, ::-1]  
        dst = k2[matches[:, 1]][:, ::-1]
        disp = dst - src
        angles = np.arctan2(disp[:, 1], disp[:, 0])
        angle_std = float(np.std(angles)) if angles.size > 0 else 0.0
    except Exception:
        angle_std = 0.0

    logging.info(f"Debug SIFT: Angle std = {angle_std:.4f} rad, Matches = {len(matches)}")
    return None, k1, k2, matches, angle_std

def match_features_dense_sift(img1, img2, method="dense_sift", step=8, size=64, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, ratio_test=0.75):
    
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
    except Exception:
        try:
            sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
        except Exception:
            logging.error("SIFT is not available in your OpenCV build.")
            return None, np.array([[kp.pt[1], kp.pt[0]] for kp in []]), np.array([[kp.pt[1], kp.pt[0]] for kp in []]), []

    def dense_keypoints(im, step, size):
        keypoints = []
        for y in range(step // 2, im.shape[0] - step // 2, step):
            for x in range(step // 2, im.shape[1] - step // 2, step):
                keypoints.append(cv2.KeyPoint(float(x), float(y), size))
        return keypoints

    kp1 = dense_keypoints(img1, step, size)
    kp2 = dense_keypoints(img2, step, size)

    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)

    if des1 is None or des2 is None or len(des1) < min_matches or len(des2) < min_matches:
        return None, np.array([[kp.pt[1], kp.pt[0]] for kp in kp1]), np.array([[kp.pt[1], kp.pt[0]] for kp in kp2]), []

    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    raw_matches = bf.match(des1, des2)
    
    raw_matches = sorted(raw_matches, key=lambda x: x.distance)
    
    matches = np.array([[m.queryIdx, m.trainIdx] for m in raw_matches])

    
    k1 = np.array([[kp.pt[1], kp.pt[0]] for kp in kp1])
    k2 = np.array([[kp.pt[1], kp.pt[0]] for kp in kp2])

    if len(matches) < min_matches:
        return None, k1, k2, matches

    
    src = k1[matches[:, 0]][:, ::-1]
    dst = k2[matches[:, 1]][:, ::-1]

    disp = dst - src
    angles = np.arctan2(disp[:, 1], disp[:, 0])
    angle_std = float(np.std(angles))
    logging.info(f"Debug {method.upper()}: Angle std = {angle_std:.4f} rad, Matches = {len(matches)}")

    return None, k1, k2, matches, angle_std

def match_features_orb(img1, img2, method="orb", nfeatures=2000, scaleFactor=1.2, nlevels=8, fastThreshold=15, ratio_test=0.68):
    
    try:
        a1 = (img1 * 255.0).clip(0, 255).astype(np.uint8) if getattr(img1, 'dtype', None) is not None and img1.dtype.kind == 'f' else img1.astype(np.uint8)
        a2 = (img2 * 255.0).clip(0, 255).astype(np.uint8) if getattr(img2, 'dtype', None) is not None and img2.dtype.kind == 'f' else img2.astype(np.uint8)
    except Exception:
        a1 = img1.astype(np.uint8)
        a2 = img2.astype(np.uint8)

    
    if a1.ndim == 3:
        if a1.shape[2] == 4:
            try:
                a1 = cv2.cvtColor(a1, cv2.COLOR_RGBA2GRAY)
            except Exception:
                a1 = cv2.cvtColor(a1[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            a1 = cv2.cvtColor(a1, cv2.COLOR_RGB2GRAY)
    if a2.ndim == 3:
        if a2.shape[2] == 4:
            try:
                a2 = cv2.cvtColor(a2, cv2.COLOR_RGBA2GRAY)
            except Exception:
                a2 = cv2.cvtColor(a2[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            a2 = cv2.cvtColor(a2, cv2.COLOR_RGB2GRAY)

    
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        a1 = clahe.apply(a1)
        a2 = clahe.apply(a2)
    except Exception:
        pass

    
    try:
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
                             edgeThreshold=31, patchSize=31, fastThreshold=fastThreshold,
                             WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE)
    except Exception:
        
        orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(a1, None)
    kp2, des2 = orb.detectAndCompute(a2, None)

    
    k1 = np.array([[kp.pt[1], kp.pt[0]] for kp in kp1]) if kp1 is not None else np.empty((0, 2))
    k2 = np.array([[kp.pt[1], kp.pt[0]] for kp in kp2]) if kp2 is not None else np.empty((0, 2))

    if des1 is None or des2 is None or len(k1) < min_matches or len(k2) < min_matches:
        logging.info(f"Debug ORB: Not enough descriptors (des1={None if des1 is None else len(des1)}, des2={None if des2 is None else len(des2)}) or keypoints (k1={len(k1)}, k2={len(k2)})")
        return None, k1, k2, np.empty((0, 2), dtype=int)

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw = bf.knnMatch(des1, des2, k=2)
    except Exception:
        
        raw = [ [m] for m in bf.match(des1, des2) ]

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_test * n.distance:
                good.append((m.queryIdx, m.trainIdx))
        elif len(pair) == 1:
            m = pair[0]
            good.append((m.queryIdx, m.trainIdx))

    
    if len(good) < min_matches:
        bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            cross = bf2.match(des1, des2)
            cross = sorted(cross, key=lambda x: x.distance)
            good = [(m.queryIdx, m.trainIdx) for m in cross[:max(min_matches, len(cross))]]
        except Exception:
            pass

    if not good:
        return None, k1, k2, np.empty((0, 2), dtype=int)

    matches = np.array(good, dtype=int)

    
    try:
        src = k1[matches[:, 0]][:, ::-1]  
        dst = k2[matches[:, 1]][:, ::-1]
        disp = dst - src
        angles = np.arctan2(disp[:, 1], disp[:, 0])
        angle_std = float(np.std(angles)) if angles.size > 0 else 0.0
    except Exception:
        angle_std = 0.0

    logging.info(f"Debug ORB: Matches={len(matches)}, Angle std={angle_std:.4f} rad")
    return None, k1, k2, matches, angle_std

def apply_ransac_to_matches(k1, k2, matches, model='affine', thresh=5.0, min_inliers=4):
    
    if matches is None or len(matches) == 0:
        return np.zeros(0, dtype=bool), None

    
    src_pts = np.array([k1[int(m[0])][::-1] for m in matches], dtype=np.float32)  
    dst_pts = np.array([k2[int(m[1])][::-1] for m in matches], dtype=np.float32)

    if model == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=thresh)
        if mask is None:
            return np.zeros(len(matches), dtype=bool), None
        mask = mask.ravel().astype(bool)
        
        return mask, M
    else:
        
        M2x3, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=thresh)
        if mask is None:
            return np.zeros(len(matches), dtype=bool), None
        mask = mask.ravel().astype(bool)
        M = None
        if M2x3 is not None:
            M = np.eye(3, dtype=np.float32)
            M[:2, :] = M2x3
        
        if np.count_nonzero(mask) < min_inliers:
            return np.zeros(len(matches), dtype=bool), None
        return mask, M

def save_ransac_text(fname1, fname2, M, inlier_mask, thresh, model='affine', accepted=True):
    
    try:
        dir_type = "accepted" if accepted else "rejected"
        base = os.path.join(MATCH_VIS_DIR, "ransac_models", dir_type)
        os.makedirs(base, exist_ok=True)
        
        if accepted and (model is not None) and str(model).lower().startswith('aff'):
            out_name = "ransac_affine_accepted.txt"
            out_path = os.path.join(base, out_name)
            file_mode = 'a'
            write_separator = True
        else:
            out_name = f"ransac_{os.path.splitext(fname1)[0]}_{os.path.splitext(fname2)[0]}.txt"
            out_path = os.path.join(base, out_name)
            file_mode = 'w'
            write_separator = False

        inliers_count = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None and len(inlier_mask) > 0 else 0
        total_matches = int(len(inlier_mask)) if inlier_mask is not None else 0

        
        with open(out_path, file_mode) as f:
            
            if write_separator:
                f.write("\n----\n\n")
            f.write(f"RANSAC result for {fname1}  <->  {fname2}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Accepted: {accepted}\n")
            f.write(f"Inliers: {inliers_count}\n")
            f.write(f"Total matches considered: {total_matches}\n")
            f.write(f"RANSAC reprojection threshold: {thresh}\n\n")

            f.write("Transformation matrix (3x3)\n")
            if M is None:
                f.write("None\n")
            else:
                
                try:
                    mat_str = np.array2string(np.asarray(M), formatter={'float_kind':lambda x: f"{x:.6g}"})
                except Exception:
                    mat_str = str(M)
                f.write(mat_str + "\n\n")

            
            if M is not None:
                try:
                    A = np.asarray(M)[:2, :2]
                    tx = float(M[0, 2])
                    ty = float(M[1, 2])
                    
                    scale_x = float(np.sqrt(A[0, 0]**2 + A[1, 0]**2))
                    scale_y = float(np.sqrt(A[0, 1]**2 + A[1, 1]**2))
                    rotation_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
                    f.write("Approximate decomposition (affine approximation):\n")
                    f.write(f"  rotation_deg: {rotation_deg:.4f}\n")
                    f.write(f"  scale_x: {scale_x:.6g}\n")
                    f.write(f"  scale_y: {scale_y:.6g}\n")
                    f.write(f"  tx: {tx:.6g}\n")
                    f.write(f"  ty: {ty:.6g}\n")
                except Exception as e:
                    f.write(f"Failed to decompose matrix: {e}\n")

        logging.info(f"Saved RANSAC info to {out_path}")
    except Exception as e:
        logging.warning(f"Failed to write RANSAC text file for {fname1} <-> {fname2}: {e}")

def save_match_vis(img1, img2, k1, k2, matches, p1, p2, accepted=True, method="sift", angle_std=0):
   
    global ransac_enabled, ransac_thresh, ransac_min_inliers

    out_dir = get_output_paths(method)[2 if accepted else 3]
    fname1 = os.path.basename(p1)  
    fname2 = os.path.basename(p2)  
    out_path = os.path.join(out_dir, f"match_{os.path.splitext(fname1)[0]}_{os.path.splitext(fname2)[0]}.png")

    
    if matches is None or len(matches) < min_matches:
        
        expl_path = os.path.join(out_dir, f"match_explanation_{os.path.splitext(fname1)[0]}_{os.path.splitext(fname2)[0]}.txt")
        safe_matches = [] if matches is None else matches
        safe_angle_std = angle_std if angle_std is not None else float('inf')
        explanation = analyze_match_decision(safe_matches, safe_angle_std, 0.0, method)
        with open(expl_path, 'w') as f:
            f.write(f"Explanation for {fname1} and {fname2}:\n{explanation}\n")
        logging.info(f"Debug {method.upper()}: Skipping visualization - not enough matches ({0 if matches is None else len(matches)} < {min_matches})")
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)
    
    pad1 = target_h - h1
    pad2 = target_h - h2
    
    v1 = float(np.nanmin(img1)) if img1.size else 0.0
    v2 = float(np.nanmin(img2)) if img2.size else 0.0
    if pad1 > 0:
        img1_p = np.pad(img1, ((0, pad1), (0, 0)), mode='constant', constant_values=v1)
    else:
        img1_p = img1
    if pad2 > 0:
        img2_p = np.pad(img2, ((0, pad2), (0, 0)), mode='constant', constant_values=v2)
    else:
        img2_p = img2

    combined = np.hstack((img1_p, img2_p))
    ax.imshow(combined, cmap='gray')
    h, w = combined.shape
    valid_indices = []
    
    ransac_pts_img1 = []
    ransac_pts_img2 = []

    
    pts1 = np.array([k1[int(m[0])][::-1] for m in matches], dtype=float)
    pts2 = np.array([k2[int(m[1])][::-1] for m in matches], dtype=float)
    
    pts2[:, 0] += img1_p.shape[1]

    
    pts1[:, 0] = np.clip(pts1[:, 0], 0, w - 1)
    pts1[:, 1] = np.clip(pts1[:, 1], 0, h - 1)
    pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
    pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)

    displacements = pts2 - pts1
    lengths = np.sqrt(np.sum(displacements**2, axis=1)).tolist()
    angles = np.arctan2(displacements[:, 1], displacements[:, 0]).tolist()

    
    inlier_mask = None
    M_ransac = None
    if ransac_enabled:
        
        
        
        if len(matches) < 4:
            logging.info(f"Debug {method.upper()}: Not enough matches ({len(matches)}) to run RANSAC — using fallback: treating all matches as RANSAC inliers.")
            inlier_mask = np.ones(len(matches), dtype=bool)
            M_ransac = None
        else:
            try:
                inlier_mask, M_ransac = apply_ransac_to_matches(k1, k2, matches, model='affine', thresh=ransac_thresh, min_inliers=ransac_min_inliers)
                logging.info(f"Debug {method.upper()}: RANSAC inliers = {np.count_nonzero(inlier_mask)} / {len(matches)}")
            except Exception as e:
                logging.info(f"Debug {method.upper()}: RANSAC failed: {e}")
                inlier_mask = None

        
        try:
            save_ransac_text(fname1, fname2, M_ransac, inlier_mask, ransac_thresh, model='affine', accepted=accepted)
        except Exception as e:
            logging.warning(f"Failed to save RANSAC text for {fname1} <-> {fname2}: {e}")

    total_lines = len(matches) if len(matches) > 0 else 1  
    green_lines = 0

    main_centroid = 0
    main_cluster_angle = 0.0

    if lengths and len(lengths) > 1:
        
        n_samples = len(lengths)
        n_clusters = min(5, n_samples)
        if n_clusters < 1:
            main_centroid = np.median(lengths) if lengths else 0
            logging.info(f"Debug {method.upper()}: Not enough samples for clustering, using median length = {main_centroid}")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(np.array(lengths).reshape(-1, 1))
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_.flatten()
            cluster_sizes = np.bincount(cluster_labels)
            main_cluster_idx = np.argmax(cluster_sizes)
            main_centroid = cluster_centers[main_cluster_idx]
            logging.info(f"Debug {method.upper()}: Cluster centroids = {cluster_centers}, Sizes = {cluster_sizes}, Main centroid = {main_centroid}")

        
        angle_vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        
        n_angle_clusters = min(8, n_samples) if n_samples > 0 else 1
        if n_angle_clusters >= 1 and len(angles) > 0:
            try:
                akmeans = KMeans(n_clusters=n_angle_clusters, random_state=42).fit(angle_vectors)
                a_labels = akmeans.labels_
                a_centers = akmeans.cluster_centers_
                a_sizes = np.bincount(a_labels)
                main_a_idx = np.argmax(a_sizes)
                
                cx, cy = a_centers[main_a_idx]
                main_cluster_angle = np.arctan2(cy, cx)
                logging.info(f"Debug {method.upper()}: Angle cluster centers (vec) = {a_centers}, sizes = {a_sizes}, main angle (deg) = {np.degrees(main_cluster_angle):.2f}")
            except Exception as e:
                logging.info(f"Debug {method.upper()}: Angle clustering failed: {e}")
                main_cluster_angle = np.median(angles) if angles else 0.0
        else:
            main_cluster_angle = np.median(angles) if angles else 0.0

        
        angle_thresh_rad = np.deg2rad(matching_angle)
        for m in range(len(matches)):
            pt1 = pts1[m]
            pt2 = pts2[m]

            disp = pt2 - pt1
            length = np.sqrt(disp[0]**2 + disp[1]**2)
            angle = np.arctan2(disp[1], disp[0])

            
            delta = np.arctan2(np.sin(angle - main_cluster_angle), np.cos(angle - main_cluster_angle))
            is_angle_valid = abs(delta) <= angle_thresh_rad

            med_angle = main_cluster_angle

            
            is_inlier = bool(inlier_mask[m]) if (inlier_mask is not None and len(inlier_mask) == len(matches)) else True

            logging.info(f"Debug {method.upper()}: Match {m} - pt1={pt1}, pt2={pt2}, disp={disp}, angle={np.degrees(angle):.2f}°, length={length:.2f}, main_centroid={main_centroid:.2f}, valid_length={abs(length - main_centroid) <= len_gap}, angle_diff_deg={np.degrees(delta):.2f}, angle_thresh_deg={matching_angle}, ransac_inlier={is_inlier}")

            if ransac_enabled and is_inlier:
                ransac_pts_img1.append(tuple(pt1))
                ransac_pts_img2.append(tuple(pt2))

            if is_angle_valid:
                if abs(length - main_centroid) <= len_gap and (not ransac_enabled or is_inlier):  
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=0.5)  
                    valid_indices.append(m)
                    green_lines += 1
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'm-', linewidth=0.5)  
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=0.5)  

    else:
        
        main_centroid = np.median(lengths) if lengths else 0
        main_cluster_angle = np.median(angles) if angles else 0.0
        angle_thresh_rad = np.deg2rad(matching_angle)
        for m in range(len(matches)):
            pt1 = pts1[m]
            pt2 = pts2[m]

            disp = pt2 - pt1
            length = np.sqrt(disp[0]**2 + disp[1]**2)
            angle = np.arctan2(disp[1], disp[0])
            delta = np.arctan2(np.sin(angle - main_cluster_angle), np.cos(angle - main_cluster_angle))
            is_angle_valid = abs(delta) <= angle_thresh_rad

            
            is_inlier = bool(inlier_mask[m]) if (inlier_mask is not None and len(inlier_mask) == len(matches)) else True

            logging.info(f"Debug {method.upper()}: (fallback) Match {m} - angle_diff_deg={np.degrees(delta):.2f}, angle_thresh_deg={matching_angle}, ransac_inlier={is_inlier}")

            if ransac_enabled and is_inlier:
                ransac_pts_img1.append(tuple(pt1))
                ransac_pts_img2.append(tuple(pt2))

            if is_angle_valid:
                if abs(length - main_centroid) <= len_gap and (not ransac_enabled or is_inlier):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=0.5)
                    valid_indices.append(m)
                    green_lines += 1
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'm-', linewidth=0.5)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=0.5)

    ratio = green_lines / total_lines
    logging.info(f"Debug {method.upper()}: Green lines ratio = {ratio:.2%}")

    
    def draw_polygon_from_points(pt_list, color='b', linewidth=1.0, alpha=0.9):
        if len(pt_list) < 2:
            return
        pts = np.array(pt_list)
        
        centroid = pts.mean(axis=0)
        angles_order = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
        order = np.argsort(angles_order)
        ordered = pts[order]
        
        poly_x = np.r_[ordered[:,0], ordered[0,0]]
        poly_y = np.r_[ordered[:,1], ordered[0,1]]
        ax.plot(poly_x, poly_y, color=color, linewidth=linewidth, alpha=alpha)
        
        ax.scatter(ordered[:,0], ordered[:,1], c=color, s=10, alpha=alpha)

    if ransac_enabled and ransac_pts_img1:
        
        draw_polygon_from_points(ransac_pts_img1, color='b', linewidth=1.0, alpha=0.9)
    if ransac_enabled and ransac_pts_img2:
        draw_polygon_from_points(ransac_pts_img2, color='b', linewidth=1.0, alpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    if valid_indices:
        valid_displacements = np.array(displacements)[valid_indices]
        valid_angles = np.arctan2(valid_displacements[:, 1], valid_displacements[:, 0])
        med_angle = np.median(valid_angles) if valid_angles.size > 0 else 0
        avg_length = np.mean([lengths[i] for i in valid_indices]) if valid_indices else 0
        logging.info(f"Debug {method.upper()}: Accepted - Avg length = {avg_length:.2f}, Median angle = {np.degrees(med_angle):.2f}°")
    else:
        med_angle = 0
        avg_length = 0
        valid_angles = np.array([])
        logging.info(f"Debug {method.upper()}: No valid (green) lines found")

    
    
    if valid_angles.size > 0:
        angle_std_for_decision = float(np.std(valid_angles))
    elif len(angles) > 0:
        angle_std_for_decision = float(np.std(angles))
    else:
        angle_std_for_decision = float('inf')

    
    if method == selected_method and valid_indices and accepted:
        offset_file = os.path.join(MATCH_VIS_DIR, "match_offsets.txt")
        with open(offset_file, 'a') as f:
            f.write(f"{fname1} {fname2} {avg_length:.2f} {np.degrees(med_angle):.2f}\n")
        logging.info(f"Debug {method.upper()}: Written to {offset_file}: {fname1} {fname2} {avg_length:.2f} {np.degrees(med_angle):.2f}")

    
    expl_path = os.path.join(out_dir, f"match_explanation_{os.path.splitext(fname1)[0]}_{os.path.splitext(fname2)[0]}.txt")
    explanation = analyze_match_decision(matches, angle_std_for_decision, ratio, method)
    with open(expl_path, 'w') as f:
        f.write(f"Explanation for {fname1} and {fname2}:\n{explanation}\n")

    
    if angle_std_for_decision <= np.deg2rad(matching_angle) and ratio >= matching_ratio:
        if displacements is not None and getattr(displacements, 'size', 0) > 0:
            med_disp = np.median(displacements, axis=0)
        else:
            med_disp = np.array([0.0, 0.0])
        M = np.eye(3)
        M[0, 2] = med_disp[0]
        M[1, 2] = med_disp[1]
        return M
    return None

def level_tile_by_3points(tile, points=None):
    
    h, w = tile.shape
    if points is None:
        
        points = [(0, 0), (w - 1, 0), (0, h - 1)]

    if len(points) != 3:
        raise ValueError("Exactly three points required for 3-point leveling")

    pts = []
    vals = []
    for (x, y) in points:
        
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        pts.append([xi, yi, 1.0])
        vals.append(tile[yi, xi])

    A = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)  
    b = np.array(vals, dtype=np.float64).reshape(3, 1)

    
    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return tile.copy(), (0.0, 0.0, 0.0)
    a, b_par, c = params.flatten().tolist()

    
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    plane = a * X + b_par * Y + c
    leveled = tile - plane
    return leveled, (a, b_par, c)

def replace_tiff_with_gwy(file_path):
    gwy_folder = os.path.join(TILE_DIR, "gwy")
    os.makedirs(gwy_folder, exist_ok=True)
    match_gwy = os.path.join(TILE_DIR,"gwy/match_offsets.txt")
    if os.path.exists(match_gwy):
        os.remove(match_gwy)
    
    shutil.copy(file_path, match_gwy) 
    
    try:
        
        with open(match_gwy, 'r') as file:
            content = file.read()

        
        updated_content = content.replace('.png', '.gwy')
        updated_content = content.replace('.tiff', '.gwy')
        
        with open(match_gwy, 'w') as file:
            file.write(updated_content)

        print(f"Replaced all '.tiff' with '.gwy' in {match_gwy}")

    except FileNotFoundError:
        print(f"Error: The file {match_gwy} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

def main(methods, tile_dir=None, matching_ratio_param=None, matching_angle_param=None, min_matches_param=None, len_gap_param=None, selected_method_param=None, three_point_level_param=False, ransac_enabled_param=False, ransac_thresh_param=5.0, ransac_min_inliers_param=4, preprocess_enabled_param=True):
   
    global TILE_DIR, MATCH_VIS_DIR, matching_ratio, matching_angle, min_matches, len_gap, selected_method, ransac_enabled, ransac_thresh, ransac_min_inliers, preprocess_enabled

    if tile_dir is not None:
        TILE_DIR = tile_dir
    if matching_ratio_param is not None:
        matching_ratio = matching_ratio_param
    if matching_angle_param is not None:
        matching_angle = matching_angle_param
    if min_matches_param is not None:
        min_matches = min_matches_param
    if len_gap_param is not None:
        len_gap = len_gap_param
    if selected_method_param is not None:
        selected_method = selected_method_param
    if ransac_enabled_param is not None:
        ransac_enabled = ransac_enabled_param
    if ransac_thresh_param is not None:
        ransac_thresh = ransac_thresh_param
    if ransac_min_inliers_param is not None:
        ransac_min_inliers = ransac_min_inliers_param
    if preprocess_enabled_param is not None:
        preprocess_enabled = preprocess_enabled_param

    
    MATCH_VIS_DIR = os.path.join(TILE_DIR, "match_vis_skimage")
    shutil.rmtree(MATCH_VIS_DIR, ignore_errors=True)
    os.makedirs(MATCH_VIS_DIR, exist_ok=True)
    
   
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(MATCH_VIS_DIR,"stitching_log.log")),
            logging.StreamHandler()
        ]
    )
    tiles, raw_tiles, h, w, paths = load_tiles(TILE_DIR)

    
    if three_point_level_param:
        logging.info("Applying per-tile 3-point leveling to raw tiles (Gwyddion-like)")
        for idx in range(len(raw_tiles)):
            try:
                leveled, plane = level_tile_by_3points(raw_tiles[idx], points=None)
                raw_tiles[idx] = leveled
                
                tiles[idx] = preprocess_image(raw_tiles[idx])
                logging.info(f"Leveled tile {os.path.basename(paths[idx])} with plane params a={plane[0]:.6g}, b={plane[1]:.6g}, c={plane[2]:.6g}")
            except Exception as e:
                logging.warning(f"Failed to level tile {paths[idx]}: {e}")

    
    offset_file = os.path.join(MATCH_VIS_DIR, "match_offsets.txt")
    if os.path.exists(offset_file):
        os.remove(offset_file)

    for method in methods:
        logging.info(f"\nProcessing with {method.upper()} method")
        OUTPUT_IMAGE_COMPONENT, OUTPUT_IMAGE_ALL, MATCH_VIS_ACCEPTED, MATCH_VIS_REJECTED = get_output_paths(method)

        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                p1, p2 = paths[i], paths[j]
                fname1 = os.path.basename(p1)  
                fname2 = os.path.basename(p2)  

                match_func = {
                     "sift": match_features_sift,
                      "orb": match_features_orb,
                     "dense_sift": match_features_dense_sift,  
                }[method]

                
                res = match_func(tiles[i], tiles[j], method=method)
                
                if res is None:
                    M = None; k1 = k2 = matches = angle_std = None
                elif isinstance(res, (list, tuple)):
                    if len(res) == 5:
                        M, k1, k2, matches, angle_std = res
                    elif len(res) == 4:
                        M, k1, k2, matches = res
                        angle_std = 0
                    else:
                        
                        raise ValueError(f"match_func returned unexpected number of values: {len(res)}")
                else:
                    raise ValueError("match_func returned unsupported type")

                logging.info(f" {method.upper()}: Found {len(matches) if matches is not None else 0} matches between {fname1} and {fname2}")

                if M is None:
                    M = save_match_vis(tiles[i], tiles[j], k1, k2, matches, p1, p2, accepted=False, method=method, angle_std=angle_std)
                if M is not None:
                    save_match_vis(tiles[i], tiles[j], k1, k2, matches, p1, p2, accepted=True, method=method, angle_std=angle_std)
    replace_tiff_with_gwy(os.path.join(MATCH_VIS_DIR, "match_offsets.txt"))
if __name__ == "__main__":
    
    TILE_DIR = "/Volumes/T7/last/after_0.5T/0V_after_strain/ErrorSignal_Backward"  # Folder with tiles
    MATCH_VIS_DIR = os.path.join(TILE_DIR, "match_vis_skimage")
    matching_ratio = 0.3 #Percentage of correct (green) lines
    matching_angle = 10 # Maximum deviation angle in degrees
    min_matches = 4  # Reduced for testing
    len_gap = 10  # Maximum distance from the cluster centroid for valid lines
    selected_method = "sift"  # Selecting a method for calculating average values
    ransac_enabled = True  # Enable RANSAC
    ransac_thresh = 5.0  # RANSAC reprojection threshold
    ransac_min_inliers = 4  # Minimum inliers for RANSAC  
    
    
    main(methods=["sift"], tile_dir=TILE_DIR, matching_ratio_param=matching_ratio, matching_angle_param=matching_angle, min_matches_param=min_matches, len_gap_param=len_gap, selected_method_param=selected_method, three_point_level_param=False, ransac_enabled_param=ransac_enabled, ransac_thresh_param=ransac_thresh, ransac_min_inliers_param=ransac_min_inliers, preprocess_enabled_param=False)