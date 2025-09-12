
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import map_coordinates
import json
import sys
import math
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

try:
    import gwyfile  
except Exception:
    gwyfile = None


DEBUG = False
TILE_DIR = None   
BASE_FILENAME = "FeS"          
CENTER_CROP = None                            

MATCH_VIS_DIR = None  

TILE_SIZE_UM = 50


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def setup_logging(log_dir=None, filename=None, maxBytes=5*1024*1024, backupCount=5):
    
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        script_dir = os.getcwd()
    if log_dir is None:
        log_dir = script_dir
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        
        log_dir = os.getcwd()
    if filename is None:
        filename = (BASE_FILENAME or 'attocube') + '.log'
    log_path = os.path.join(log_dir, filename)
    
    try:
        if os.path.exists(log_path):
            
            os.remove(log_path)
    except Exception:
        
        print(f"Warning: could not remove existing log file {log_path}")
    try:
        handler = RotatingFileHandler(log_path, maxBytes=maxBytes, backupCount=backupCount, encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(handler)
        
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO if not DEBUG else logging.DEBUG)
        logger.addHandler(sh)
        logger.info(f'Logging initialized to {log_path}')
    except Exception:
        
        logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
        logger.warning('Failed to initialize RotatingFileHandler, falling back to basic logging')

    
    class StreamToLogger:
        """File-like object that redirects writes to a logging.Logger instance."""
        def __init__(self, logger_obj, level=logging.INFO):
            self.logger = logger_obj
            self.level = level
            self._buffer = ''

        def write(self, buf):
            if not buf:
                return
            try:
                s = str(buf)
            except Exception:
                return
            
            for line in s.rstrip('\n').splitlines():
                if line.strip():
                    try:
                        self.logger.log(self.level, line)
                    except Exception:
                        
                        sys.__stderr__.write(line + '\n')

        def flush(self):
            try:
                pass
            except Exception:
                pass

    try:
        
        logging.captureWarnings(True)
    except Exception:
        pass

    
    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
    except Exception:
        
        logger.exception('Failed to redirect stdout/stderr to logger')

    
    try:
        original_excepthook = sys.excepthook
        def _handle_exception(exc_type, exc_value, exc_tb):
            
            if exc_type is KeyboardInterrupt:
                original_excepthook(exc_type, exc_value, exc_tb)
                return
            logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_tb))
            try:
                original_excepthook(exc_type, exc_value, exc_tb)
            except Exception:
                pass
        sys.excepthook = _handle_exception
    except Exception:
        logger.exception('Failed to install excepthook')

    return log_path





def _nice_gwy_channel_name(fname: str, key: str, idx: int) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    
    try:
        part = key.strip('/').split('/')[0]
        return f"{base} [ch {part}]"
    except Exception:
        return f"{base} [ch {idx}]"


def load_gwy_tiles(tile_dir, center_crop_size=None):
    
    if gwyfile is None:
        raise RuntimeError("gwyfile is not installed. Install with: pip install gwyfile")
    names = [f for f in os.listdir(tile_dir) if f.lower().endswith('.gwy')]
    names.sort()
    if not names:
        raise RuntimeError(f"No GWY files in {tile_dir}")

    channel_mats = {}
    channel_metadata = {}
    tile_names = {}
    sizes = {}

    
    
    mats_per_channel = {}
    meta_per_channel = {}

    h = w = None
    global_index = 0

    for fname in names:
        fp = os.path.join(tile_dir, fname)
        try:
            gwy = gwyfile.load(fp)
        except Exception as e:
            if DEBUG:
                print(f"Skip (read error): {fname}: {e}")
            continue

        
        data_keys = [k for k in gwy.keys() if k.endswith('/data')]
        data_keys.sort()
        if not data_keys:
            if DEBUG:
                print(f"No data fields in {fname}")
            continue

        for di, key in enumerate(data_keys):
            df = gwy[key]
            
            arr = getattr(df, 'data', None)
            if arr is None:
                
                if hasattr(df, '__array__'):
                    arr = np.array(df)
                else:
                    if DEBUG:
                        print(f"Key {key} in {fname} has no data array; skipping")
                    continue
            arr = np.asarray(arr)
            if arr.ndim != 2:
                
                continue

            xres = int(getattr(df, 'xres', arr.shape[1]))
            yres = int(getattr(df, 'yres', arr.shape[0]))
            
            raw_xreal = getattr(df, 'xreal', np.nan)
            raw_yreal = getattr(df, 'yreal', np.nan)

            
            unit_xy = None
            try:
                si_xy = getattr(df, 'si_unit_xy', None)
                if si_xy is not None:
                    unit_xy = getattr(si_xy, 'unitstr', None) or getattr(si_xy, 'symbol', None)
            except Exception:
                unit_xy = None
            if not unit_xy:
                
                unit_xy = 'm'

            def unit_multiplier(u: str) -> float:
                if not u:
                    return 1.0
                s = str(u).strip().lower()
                s = s.replace('\u00b5', 'u')  
                s = s.replace('µ', 'u')
                
                if 'm' == s or s == 'meter' or s == 'metre':
                    return 1.0
                if s in ('cm', 'centimeter', 'centimetre'):
                    return 1e-2
                if s in ('mm', 'millimeter', 'millimetre'):
                    return 1e-3
                if 'u' in s or 'um' in s or 'micron' in s or 'micrometer' in s:
                    return 1e-6
                if 'nm' in s or 'nanometer' in s:
                    return 1e-9
                if 'ang' in s or 'a' == s:
                    return 1e-10
                
                return 1.0

            try:
                mult = unit_multiplier(unit_xy)
                xreal_m = float(raw_xreal) * float(mult) if not (raw_xreal is None or (isinstance(raw_xreal, float) and np.isnan(raw_xreal))) else np.nan
                yreal_m = float(raw_yreal) * float(mult) if not (raw_yreal is None or (isinstance(raw_yreal, float) and np.isnan(raw_yreal))) else np.nan
            except Exception:
                xreal_m = np.nan; yreal_m = np.nan

            
            unit_z = None
            try:
                si_z = getattr(df, 'si_unit_z', None)
                if si_z is not None:
                    unit_z = getattr(si_z, 'unitstr', None) or getattr(si_z, 'symbol', None)
            except Exception:
                pass
            if not unit_z:
                unit_z = 'a.u.'
            unit_xy = unit_xy or 'm'

            
            img = arr
            if center_crop_size is not None:
                th, tw = center_crop_size
                ih, iw = img.shape[:2]
                sy = max(0, (ih - th)//2)
                sx = max(0, (iw - tw)//2)
                img = img[sy:sy+th, sx:sx+tw]
                yres, xres = img.shape[:2]

            ch_name = _nice_gwy_channel_name(fname, key, di)
            if ch_name not in mats_per_channel:
                mats_per_channel[ch_name] = {}
                meta_per_channel[ch_name] = {
                    'unit_z': unit_z,
                    'unit_xy': unit_xy,
                    'xreal_m': xreal_m,
                    'yreal_m': yreal_m,
                    'xres': xres,
                    'yres': yres,
                }

            mats_per_channel[ch_name][(0, global_index)] = img
            tile_names[(0, global_index)] = f"{fname}:{key}"
            sizes[f"{fname}:{key}"] = img.shape[:2]
            if h is None:
                h, w = img.shape[:2]
            global_index += 1

    if not mats_per_channel:
        raise RuntimeError("No usable data fields found in GWY files.")

    
    path_matrix = np.arange(global_index).reshape(1, -1)
    combined = {}
    for ch_dict in mats_per_channel.values():
        for k, v in ch_dict.items():
            combined[k] = v

    
    single_channel_name = 'All Tiles'
    channel_mats[single_channel_name] = combined

    
    if meta_per_channel:
        
        chosen = None
        for m in meta_per_channel.values():
            try:
                if m.get('xreal_m') is not None and m.get('xres'):
                    chosen = dict(m)
                    break
            except Exception:
                continue
        if chosen is None:
            
            chosen = dict(next(iter(meta_per_channel.values())))
        
        try:
            xr = chosen.get('xreal_m', None)
            yr = chosen.get('yreal_m', None)
            xres = chosen.get('xres', None)
            yres = chosen.get('yres', None)
            if xr is not None and xres:
                pixel_size_m_x = float(xr) / float(xres)
                chosen['pixel_size_m'] = pixel_size_m_x
                chosen['pixel_size_um'] = pixel_size_m_x * 1e6
                chosen['tile_size_um_x'] = float(xr) * 1e6
            else:
                chosen['pixel_size_m'] = None
                chosen['pixel_size_um'] = None
                chosen['tile_size_um_x'] = None
            if yr is not None and yres:
                pixel_size_m_y = float(yr) / float(yres)
                
                if chosen.get('pixel_size_m') is None:
                    chosen['pixel_size_m'] = pixel_size_m_y
                    chosen['pixel_size_um'] = pixel_size_m_y * 1e6
                chosen['pixel_size_m_y'] = pixel_size_m_y
                chosen['tile_size_um_y'] = float(yr) * 1e6
            else:
                chosen.setdefault('pixel_size_m_y', None)
                chosen.setdefault('tile_size_um_y', None)
        except Exception:
            chosen.setdefault('pixel_size_m', None)
            chosen.setdefault('pixel_size_um', None)
            chosen.setdefault('tile_size_um_x', None)
            chosen.setdefault('tile_size_um_y', None)
        first_meta = chosen
    else:
        first_meta = {'unit_z': 'a.u.', 'unit_xy': 'm', 'xreal_m': None, 'yreal_m': None, 'xres': None, 'yres': None, 'pixel_size_m': None, 'pixel_size_um': None, 'tile_size_um_x': None, 'tile_size_um_y': None}
    channel_metadata[single_channel_name] = first_meta

    all_files = names
    return channel_mats, path_matrix, h, w, tile_names, sizes, all_files, channel_metadata




def interactive_tiff_view(channel_matrices, path_matrix, height, width, output_dir, base_filename, channel_metadata=None, tile_names=None):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    import matplotlib.cm as cm
    ch_meta = channel_metadata or {}
    tile_names = tile_names or {}

    
    

    def channel_unit(_ch_name: str) -> str:
        meta = ch_meta.get(_ch_name, {})
        return meta.get('unit_z', 'a.u.')

    def auto_minmax(arr: np.ndarray, lo=2.0, hi=98.0):
        lo_v = float(np.percentile(arr, lo))
        hi_v = float(np.percentile(arr, hi))
        if hi_v <= lo_v:
            lo_v = float(np.min(arr))
            hi_v = float(np.max(arr))
        return lo_v, hi_v

    
    channels = list(channel_matrices.keys())  
    selected = channels[0]
    offsets = {(r, c): [0, 0] for r in range(path_matrix.shape[0]) for c in range(path_matrix.shape[1])}
    order = list(offsets.keys())
    active = None
    margin = 0  

    
    zoom = 1.0
    min_zoom = 0.1
    max_zoom = 8.0
    zoom_step = 1.15  
    
    cur_img_w = None
    cur_img_h = None

    
    base_pos = {}  

    def compute_base_positions_for_channel(ch):
        nonlocal base_pos, order
        logger.debug(f"compute_base_positions_for_channel: computing base positions for channel '{ch}'")
        
        matches = []
        
        
        
        candidate_dirs = []
        try:
            if MATCH_VIS_DIR:
                candidate_dirs.append(MATCH_VIS_DIR)
        except Exception:
            pass
        try:
            
            if 'folder_var' in locals() or 'folder_var' in globals():
                fv = folder_var.get() if hasattr(folder_var, 'get') else str(folder_var)
                if fv:
                    candidate_dirs.append(fv)
                    
                    candidate_dirs.append(os.path.dirname(fv))
        except Exception:
            pass
        try:
            candidate_dirs.append(os.getcwd())
        except Exception:
            pass
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_dirs.append(os.path.join(script_dir, 'old'))
            candidate_dirs.append(os.path.join(script_dir, '..', 'old'))
        except Exception:
            pass

        mo_path = None
        for d in candidate_dirs:
            try:
                if not d:
                    continue
                
                for fname in ('match_offsets.txt', 'match_offset.txt'):
                    p = os.path.join(d, fname)
                    if os.path.exists(p):
                        mo_path = p
                        break
                if mo_path:
                    break
            except Exception:
                continue

        if mo_path is None:
            
            logger.debug('compute_base_positions_for_channel: no match_offsets file found in candidate locations')
            base_pos = {}
            order = list(offsets.keys())
            return

        logger.info(f'compute_base_positions_for_channel: using match offsets file: {mo_path}')
        try:
            with open(mo_path, 'r') as mf:
                for ln in mf:
                    s = ln.strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) >= 4:
                        try:
                            f1, f2 = parts[0], parts[1]
                            length = float(parts[2])
                            angle = float(parts[3])
                            matches.append((f1, f2, length, angle))
                        except Exception:
                            continue
        except Exception:
            matches = []
        
        if not matches:
            base_pos = {}
            order = list(offsets.keys())
            return

        
        positions = {}
        
        ref_fname = matches[0][0]
        positions[ref_fname] = np.array([0.0, 0.0], dtype=float)
        
        tile_h, tile_w = height, width

        
        for fname1, fname2, length, angle in matches:
            if fname1 in positions and fname2 not in positions:
                cur = positions[fname1]
                base_x = cur[0] + tile_w
                base_y = cur[1]
                dx = -length * math.cos(math.radians(angle))
                dy = -length * math.sin(math.radians(angle))
                positions[fname2] = np.array([base_x + dx, base_y + dy], dtype=float)

        
        updated = True
        while updated:
            updated = False
            for fname1, fname2, length, angle in matches:
                if fname2 in positions and fname1 not in positions:
                    cur = positions[fname2]
                    dx =  length * math.cos(math.radians(angle))
                    dy =  length * math.sin(math.radians(angle))
                    new_x = cur[0] - tile_w + dx
                    new_y = cur[1] + dy
                    positions[fname1] = np.array([new_x, new_y], dtype=float)
                    updated = True

        
        for _ in range(1):
            for fname1, fname2, length, angle in matches:
                if fname1 in positions and fname2 not in positions:
                    cur = positions[fname1]
                    base_x = cur[0] + tile_w
                    base_y = cur[1]
                    dx = -length * math.cos(math.radians(angle))
                    dy = -length * math.sin(math.radians(angle))
                    positions[fname2] = np.array([base_x + dx, base_y + dy], dtype=float)
                elif fname2 in positions and fname1 not in positions:
                    cur = positions[fname2]
                    dx =  length * math.cos(math.radians(angle))
                    dy =  length * math.sin(math.radians(angle))
                    positions[fname1] = np.array([cur[0] - tile_w + dx, cur[1] + dy], dtype=float)

        
        min_x = min(p[0] for p in positions.values())
        min_y = min(p[1] for p in positions.values())

        
        base_pos = {}
        for (r, c) in offsets.keys():
            key = (r, c)
            tname = tile_names.get(key)
            if not tname:
                continue
            
            fname_only = tname.split(':', 1)[0]
            pos = positions.get(fname_only)
            if pos is None:
                
                continue
            x_off = int(pos[0] - min_x + margin)
            y_off = int(pos[1] - min_y + margin)
            base_pos[key] = [x_off, y_off]

        
        order = list(offsets.keys())
        return

    
    ch_state = {ch: {
        "align_rows_h": False, "align_rows_v": False,
        "align_diff_h": False, "align_diff_v": False,
        "level_points": [], "level_coeffs": None,
        
        "flip_h": False, "flip_v": False, "rot90": 0
    } for ch in channels}

    
    root = tk.Tk(); root.withdraw()
    viewer = tk.Toplevel(root); viewer.title("GWY Viewer")
    controls = tk.Toplevel(root); controls.title("Controls (GWY)")
    mode = tk.StringVar(root, value="none")  

    
    folder_var = tk.StringVar(value=str(TILE_DIR or ''))

    def browse_folder():
        
        path = filedialog.askdirectory(initialdir=(folder_var.get() or os.getcwd()))
        if path:
            folder_var.set(path)
            on_folder_change(path)

    def choose_folder_native():
        
        if sys.platform == 'darwin':
            try:
                cmd = ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select Tiles Folder")']
                res = subprocess.run(cmd, capture_output=True, text=True)
                path = res.stdout.strip()
                if path:
                    folder_var.set(path)
                    on_folder_change(path)
                    return
            except Exception:
                pass
        
        if sys.platform.startswith('linux'):
            try:
                import shutil
                if shutil.which('zenity'):
                    cmd = ['zenity', '--file-selection', '--directory', '--title=Select Tiles Folder']
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    path = res.stdout.strip()
                    if path:
                        folder_var.set(path)
                        on_folder_change(path)
                        return
            except Exception:
                pass
        
        browse_folder()

    def open_folder_native():
        
        p = folder_var.get()
        if not p:
            print("No folder set")
            return
        if not os.path.isdir(p):
            print(f"Not a directory: {p}")
            return
        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', p], check=False)
            elif sys.platform.startswith('win'):
                
                try:
                    os.startfile(p)
                except Exception:
                    subprocess.run(['explorer', p])
            else:
                
                import shutil
                opener = None
                for cmd in ('xdg-open', 'gio', 'gnome-open'):
                    if shutil.which(cmd):
                        opener = cmd; break
                if opener:
                    subprocess.run([opener, p], check=False)
                else:
                    
                    print(f"Open folder: {p}")
        except Exception as e:
            print(f"Failed to open folder {p}: {e}")

    def on_folder_change(new_path=None):
        
        nonlocal channel_matrices, path_matrix, height, width, tile_names, ch_meta
        p = new_path or folder_var.get()
        if not p or not os.path.isdir(p):
            print(f"Invalid folder: {p}")
            return
        try:
            cmats, P, h_new, w_new, tnames, sizes_new, all_files_new, ch_meta_new = load_gwy_tiles(p, CENTER_CROP)
        except Exception as e:
            print(f"Failed to load tiles from {p}: {e}")
            return
        
        try:
            channel_matrices.clear(); channel_matrices.update(cmats)
        except Exception:
            channel_matrices = cmats
        path_matrix = P
        height = h_new; width = w_new
        try:
            tile_names.clear(); tile_names.update(tnames)
        except Exception:
            tile_names = tnames
        try:
            
            ch_meta.update(ch_meta_new)
        except Exception:
            pass
        
        try:
            compute_base_positions_for_channel(selected)
        except Exception:
            pass
        
        try:
            update_slider_bounds()
        except Exception:
            pass
        try:
            update_tile_dropdown()
        except Exception:
            pass
        try:
            draw()
        except Exception:
            pass

    
    folder_frame = tk.Frame(controls)
    folder_frame.grid(row=0, column=0, columnspan=5, padx=8, pady=2, sticky='w')
    ttk.Label(folder_frame, text='Tiles folder:').pack(side='left')
    folder_entry = ttk.Entry(folder_frame, textvariable=folder_var, width=40)
    folder_entry.pack(side='left', padx=(4,6))
    btn_native = ttk.Button(folder_frame, text='Choose (Native)', command=choose_folder_native)
    btn_native.pack(side='left', padx=(4,0))
    btn_browse = ttk.Button(folder_frame, text='Browse', command=browse_folder)
    btn_browse.pack(side='left')
    btn_reload = ttk.Button(folder_frame, text='Reload', command=lambda: on_folder_change(None))
    btn_reload.pack(side='left', padx=(4,0))
    btn_open = ttk.Button(folder_frame, text='Open', command=open_folder_native)
    btn_open.pack(side='left', padx=(4,0))

    
    vmin_var = tk.StringVar(value='0')
    vmax_var = tk.StringVar(value='1')
    entry_frame = tk.Frame(controls)
    
    entry_frame.grid(row=1, columnspan=4, padx=8, pady=2, sticky='w')
    ttk.Label(entry_frame, text='Min:').pack(side='left')
    vmin_entry = ttk.Entry(entry_frame, textvariable=vmin_var, width=14)
    vmin_entry.pack(side='left', padx=(4,12))
    ttk.Label(entry_frame, text='Max:').pack(side='left')
    vmax_entry = ttk.Entry(entry_frame, textvariable=vmax_var, width=14)
    vmax_entry.pack(side='left', padx=(4,4))

    
    vmin_scale = tk.Scale(controls, from_=0, to=1, resolution=1e-9, orient=tk.HORIZONTAL, label="Min", length=200)
    vmax_scale = tk.Scale(controls, from_=0, to=1, resolution=1e-9, orient=tk.HORIZONTAL, label="Max", length=200)
    vmin_scale.grid(row=2, column=0, padx=8, pady=4, sticky='w')
    vmax_scale.grid(row=2, column=1, padx=8, pady=4, sticky='w')

    
    

    
    cmap_var = tk.StringVar(value='viridis')
    
    
    
    
    try:
        cmap_keys = [k for k in cm.cmap_d.keys() if not k.startswith('_')]
        preferred = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'gray', 'bone', 'hot', 'coolwarm', 'terrain', 'gist_earth', 'jet', 'cubehelix']
        cmap_options = []
        for p in preferred:
            if p in cmap_keys:
                cmap_options.append(p)
                cmap_keys.remove(p)
        cmap_options += sorted(cmap_keys)
    except Exception:
        cmap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray', 'hot', 'coolwarm', 'terrain', 'gist_earth', 'gwyddion']
    
    if 'gwyddion' not in cmap_options:
        cmap_options.insert(0, 'gwyddion')

    
    
    cmap_display_options = [c for c in ['viridis','plasma','inferno','magma','cividis','turbo','gray','bone','hot','coolwarm','terrain','gist_earth','jet','cubehelix','gwyddion'] if c in cmap_options or c == 'gwyddion']
    
    ttk.Label(controls, text='Colormap:').grid(row=2, column=2, padx=(8,2), pady=4, sticky='e')
    cmap_menu = ttk.Combobox(controls, textvariable=cmap_var, values=cmap_display_options, state='readonly', width=16)
    cmap_menu.grid(row=2, column=3, padx=(2,8), pady=4, sticky='w')
    
    def on_cmap_change(_=None):
        try:
            update_slider_bounds(); draw()
        except Exception:
            pass
    cmap_menu.bind('<<ComboboxSelected>>', on_cmap_change)

    
    def set_vmin_from_entry(_=None):
        s = vmin_var.get()
        val = _parse_number_with_units(s)
        if val is None:
            return
        try:
            cur_max = float(vmax_scale.get())
        except Exception:
            cur_max = val + 1.0
        if val >= cur_max:
            vmax_scale.set(val + 1e-6)
        vmin_scale.set(val)

    def set_vmax_from_entry(_=None):
        s = vmax_var.get()
        val = _parse_number_with_units(s)
        if val is None:
            return
        try:
            cur_min = float(vmin_scale.get())
        except Exception:
            cur_min = val - 1.0
        if val <= cur_min:
            vmin_scale.set(val - 1e-6)
        vmax_scale.set(val)

    vmin_entry.bind('<Return>', set_vmin_from_entry)
    vmin_entry.bind('<FocusOut>', set_vmin_from_entry)
    vmax_entry.bind('<Return>', set_vmax_from_entry)
    vmax_entry.bind('<FocusOut>', set_vmax_from_entry)

    
    def on_vmin_change(_=None):
        try:
            vmin = float(vmin_scale.get()); vmax = float(vmax_scale.get())
            if vmin >= vmax: vmin_scale.set(vmax - 1e-6)
        except Exception:
            pass
        try:
            vmin_var.set(str(float(vmin_scale.get())))
        except Exception:
            pass
        draw()

    def on_vmax_change(_=None):
        try:
            vmin = float(vmin_scale.get()); vmax = float(vmax_scale.get())
            if vmax <= vmin: vmax_scale.set(vmin + 1e-6)
        except Exception:
            pass
        try:
            vmax_var.set(str(float(vmax_scale.get())))
        except Exception:
            pass
        draw()

    vmin_scale.configure(command=lambda _val: on_vmin_change())
    vmax_scale.configure(command=lambda _val: on_vmax_change())

    
    btn_frame = tk.Frame(controls); btn_frame.grid(row=3, column=0, columnspan=5, sticky='nsew')
    
    controls.grid_rowconfigure(2, weight=1)
    controls.grid_columnconfigure(0, weight=0)
    controls.grid_columnconfigure(1, weight=0)
    scale_enabled_var = tk.BooleanVar(value=False)
    scale_width_var = tk.StringVar(value=str(50))   
    scale_label_var = tk.StringVar(value='50 µm')

    scale_frame = tk.Frame(btn_frame)
    scale_frame.pack(fill='x', padx=6, pady=4)
    scale_cb = ttk.Checkbutton(scale_frame, text='Show scale bar', variable=scale_enabled_var)
    scale_cb.pack(side='left')
    ttk.Label(scale_frame, text='Width (µm):').pack(side='left')
    scale_w_entry = ttk.Entry(scale_frame, textvariable=scale_width_var, width=8)
    scale_w_entry.pack(side='left')
    ttk.Label(scale_frame, text='Label:').pack(side='left', padx=(8,2))
    scale_label_entry = ttk.Entry(scale_frame, textvariable=scale_label_var, width=10)
    scale_label_entry.pack(side='left')
    

    
    tile_size_var = tk.StringVar(value=str(TILE_SIZE_UM if TILE_SIZE_UM is not None else ''))
    unit_mode_var = tk.StringVar(value='auto')
    tile_frame = tk.Frame(btn_frame)
    tile_frame.pack(fill='x', padx=6, pady=4)
    ttk.Label(tile_frame, text='Tile size (µm):').pack(side='left')
    tile_size_entry = ttk.Entry(tile_frame, textvariable=tile_size_var, width=8)
    tile_size_entry.pack(side='left')
    btn_apply_tile = ttk.Button(tile_frame, text='Apply tile size')
    btn_apply_tile.pack(side='left', padx=6)
    ttk.Label(tile_frame, text='Height mode:').pack(side='left', padx=(8,2))
    unit_menu = ttk.Combobox(tile_frame, textvariable=unit_mode_var, values=['auto','deg','height'], state='readonly', width=8)
    unit_menu.pack(side='left')
    btn_apply_unit = ttk.Button(tile_frame, text='Apply unit')
    btn_apply_unit.pack(side='left', padx=6)

    
    tile_select_var = tk.StringVar(value='')
    
    tile_name_map = {}
    def update_tile_dropdown():
        
        vals = []
        tile_name_map.clear()
        try:
            
            if isinstance(tile_names, dict):
                for k, nm in tile_names.items():
                    try:
                        r, c = k if (isinstance(k, (list, tuple)) and len(k) == 2) else map(int, str(k).split(','))
                    except Exception:
                        
                        continue
                    label = f"{r},{c}: {nm}"
                    vals.append(label)
                    tile_name_map[label] = (r, c)
            else:
                
                source = None
                if 'order' in locals() or 'order' in globals():
                    source = order
                else:
                    try:
                        first_ch = channels[0]
                        source = list(channel_matrices[first_ch].keys())
                    except Exception:
                        source = []
                for rc in source:
                    try:
                        r, c = rc
                    except Exception:
                        continue
                    label = f"{r},{c}"
                    vals.append(label)
                    tile_name_map[label] = (r, c)
        except Exception:
            vals = []
            tile_name_map.clear()
        try:
            tile_select_cb.config(values=vals)
            
            if tile_select_var.get() not in vals:
                tile_select_var.set('')
        except Exception:
            pass

    tile_select_frame = tk.Frame(btn_frame)
    tile_select_frame.pack(fill='x', padx=6, pady=4)
    ttk.Label(tile_select_frame, text='Select tile:').pack(side='left')
    tile_select_cb = ttk.Combobox(tile_select_frame, textvariable=tile_select_var, values=[], state='readonly', width=30)
    tile_select_cb.pack(side='left', padx=(6,2))
    def on_tile_dropdown_change(_=None):
        nonlocal active
        lbl = tile_select_var.get()
        if not lbl:
            return
        rc = tile_name_map.get(lbl)
        if rc is None:
            
            try:
                coords = lbl.split(':', 1)[0].strip()
                r, c = map(int, coords.split(','))
                rc = (r, c)
            except Exception:
                return
        active = rc
        
        try:
            draw()
        except Exception:
            pass
    tile_select_cb.bind('<<ComboboxSelected>>', on_tile_dropdown_change)

    def apply_tile_size_for_channel(ch_name=None):
        ch_name = ch_name or selected
        meta = ch_meta.setdefault(ch_name, {})
        try:
            val = float(tile_size_var.get())
        except Exception:
            print('Invalid tile size (µm)')
            return
        
        if width and width > 0:
            px_um = float(val) / float(width)
            meta['pixel_size_um'] = px_um
            meta['tile_size_um_x'] = float(val)
            meta['tile_size_um_y'] = float(val)
            print(f"Set {ch_name} pixel_size_um = {px_um:.6g} µm/px from tile {val} µm and width {width}px")
            update_slider_bounds(); draw()
        else:
            print('Image width unknown; cannot compute µm/px')

    def apply_unit_mode_for_channel(ch_name=None):
        ch_name = ch_name or selected
        meta = ch_meta.setdefault(ch_name, {})
        mode = unit_mode_var.get()
        if mode == 'auto':
            meta['z_mode'] = 'auto'
            meta['z_mult'] = float(meta.get('z_mult', 1.0))
            
        elif mode == 'deg':
            meta['z_mode'] = 'deg'
            meta['z_mult'] = 1.0
            meta['unit_z'] = 'deg'
        elif mode == 'height':
            
            meta['z_mode'] = 'height'
            meta['z_mult'] = 1e9
            meta['unit_z'] = 'nm'
        print(f"Channel {ch_name}: set height mode '{mode}', unit_z={meta.get('unit_z')}, z_mult={meta.get('z_mult')}")
        update_slider_bounds(); draw()

    btn_apply_tile.config(command=lambda: apply_tile_size_for_channel())
    btn_apply_unit.config(command=lambda: apply_unit_mode_for_channel())

    
    crop_top_var = tk.IntVar(value=0)
    crop_bottom_var = tk.IntVar(value=0)
    crop_left_var = tk.IntVar(value=0)
    crop_right_var = tk.IntVar(value=0)

    crop_frame = tk.Frame(btn_frame)
    crop_frame.pack(fill='x', padx=6, pady=4)
    ttk.Label(crop_frame, text='Crop Top').pack(side='left')
    crop_top = tk.Scale(crop_frame, from_=0, to=0, orient='horizontal', variable=crop_top_var, length=120, command=lambda _v: on_crop_change())
    crop_top.pack(side='left')
    ttk.Label(crop_frame, text='Bottom').pack(side='left')
    crop_bottom = tk.Scale(crop_frame, from_=0, to=0, orient='horizontal', variable=crop_bottom_var, length=120, command=lambda _v: on_crop_change())
    crop_bottom.pack(side='left')

    crop_frame2 = tk.Frame(btn_frame)
    crop_frame2.pack(fill='x', padx=6, pady=0)
    ttk.Label(crop_frame2, text='Crop Left').pack(side='left')
    crop_left = tk.Scale(crop_frame2, from_=0, to=0, orient='horizontal', variable=crop_left_var, length=120, command=lambda _v: on_crop_change())
    crop_left.pack(side='left')
    ttk.Label(crop_frame2, text='Right').pack(side='left')
    crop_right = tk.Scale(crop_frame2, from_=0, to=0, orient='horizontal', variable=crop_right_var, length=120, command=lambda _v: on_crop_change())
    crop_right.pack(side='left')

    def on_crop_change(_=None):
        
        try:
            mats = channel_matrices[selected]
            big = build_image(mats)
            H, W = big.shape[:2]
        except Exception:
            return
        ct = max(0, min(crop_top_var.get(), max(0, H-1)))
        cb = max(0, min(crop_bottom_var.get(), max(0, H-1)))
        cl = max(0, min(crop_left_var.get(), max(0, W-1)))
        cr = max(0, min(crop_right_var.get(), max(0, W-1)))
        
        if ct + cb >= H:
            cb = max(0, H-1 - ct)
            crop_bottom_var.set(cb)
        if cl + cr >= W:
            cr = max(0, W-1 - cl)
            crop_right_var.set(cr)
        crop_top_var.set(ct); crop_bottom_var.set(cb); crop_left_var.set(cl); crop_right_var.set(cr)
        draw()

    
    tile_value_step_var = tk.DoubleVar(value=1.0)
    tile_value_edit_mode_var = tk.BooleanVar(value=True)
    
    tile_step_mode_var = tk.StringVar(value='display')
    value_frame = tk.Frame(btn_frame)
    value_frame.pack(fill='x', padx=6, pady=4)
    ttk.Checkbutton(value_frame, text='W/S adjust tile values', variable=tile_value_edit_mode_var).pack(side='left')
    ttk.Label(value_frame, text='Value step:').pack(side='left', padx=(8,2))
    tile_value_entry = ttk.Entry(value_frame, textvariable=tile_value_step_var, width=8)
    tile_value_entry.pack(side='left')
    ttk.Label(value_frame, text='Step units:').pack(side='left', padx=(8,2))
    step_mode_menu = ttk.Combobox(value_frame, textvariable=tile_step_mode_var, values=['display','raw'], state='readonly', width=8)
    step_mode_menu.pack(side='left')
    btn_val_inc = ttk.Button(value_frame, text='W: +val', width=8)
    btn_val_inc.pack(side='left', padx=4)
    btn_val_dec = ttk.Button(value_frame, text='S: -val', width=8)
    btn_val_dec.pack(side='left', padx=4)
    
    btn_val_reset = ttk.Button(value_frame, text='Reset tile', width=10)
    btn_val_reset.pack(side='left', padx=(8,4))

    def nudge_active_tile(dy: int):
        
        nonlocal offsets
        if active is None:
            print('No active tile selected. Click a tile first.')
            return
        off = offsets.get(active, [0, 0])
        off[1] = int(off[1]) + int(dy)
        offsets[active] = off
        draw()

    def nudge_tile_up():
        
        nudge_active_tile(-1)

    def nudge_tile_down():
        
        nudge_active_tile(1)

    def adjust_active_tile_value(delta: float):
        
        if active is None:
            print('No active tile selected. Click a tile first.')
            return
        try:
            mats = channel_matrices[selected]
            if active not in mats:
                print('Active tile has no image data')
                return
            
            meta = ch_meta.setdefault(selected, {})
            zm = float(meta.get('z_mult', 1.0) or 1.0)
            if abs(zm) < 1e-30:
                zm = 1.0
            delta_entered = float(delta)
            
            if tile_step_mode_var.get() == 'display':
                
                delta_display = delta_entered
                delta_raw = float(delta_display) / float(zm)
            else:
                
                delta_raw = float(delta_entered)
                delta_display = float(delta_raw) * float(zm)

            
            
            
            
            vmap = meta.setdefault('value_offsets', {})
            key = f"{active[0]},{active[1]}"
            prev_raw = float(vmap.get(key, 0.0))
            vmap[key] = prev_raw + float(delta_raw)
            
            vmap_display = meta.setdefault('value_offsets_display', {})
            prev_disp = float(vmap_display.get(key, 0.0))
            vmap_display[key] = prev_disp + float(delta_display)

            
            try:
                tile_arr = np.asarray(mats[active]).astype(np.float64)
                std = float(np.std(tile_arr))
                if std > 0 and abs(delta_raw) > std * 1e4:
                    print(f"Warning: applied raw delta {delta_raw} is >> tile std {std:.3e}; check unit/step-mode")
            except Exception:
                pass

            update_slider_bounds(); draw()
            print(f"Adjusted tile {active} by {delta_display} (display units) => {delta_raw} raw units")
        except Exception as e:
            print(f"Failed to adjust tile values: {e}")

    def reset_active_tile_value():
        
        if active is None:
            print('No active tile selected. Click a tile first.')
            return
        try:
            mats = channel_matrices[selected]
            if active not in mats:
                print('Active tile has no image data')
                return
            meta = ch_meta.setdefault(selected, {})
            vmap = meta.setdefault('value_offsets', {})
            key = f"{active[0]},{active[1]}"
            offset_raw = float(vmap.get(key, 0.0))
            if abs(offset_raw) < 1e-12:
                print(f'Tile {active} has no recorded value offset to reset')
                return
            
            vmap.pop(key, None)
            meta.setdefault('value_offsets_display', {}).pop(key, None)
            update_slider_bounds(); draw()
            print(f'Reset tile {active}: subtracted {offset_raw} raw units')
        except Exception as e:
            print(f'Failed to reset tile values: {e}')

    def inc_tile_value():
        try:
            step = float(tile_value_step_var.get())
        except Exception:
            step = 1.0
        adjust_active_tile_value(step)

    def dec_tile_value():
        try:
            step = float(tile_value_step_var.get())
        except Exception:
            step = 1.0
        adjust_active_tile_value(-step)

    btn_val_inc.configure(command=inc_tile_value)
    btn_val_dec.configure(command=dec_tile_value)
    btn_val_reset.configure(command=reset_active_tile_value)

    btn_finish = ttk.Button(btn_frame, text='Finish')
    
    btn_front  = ttk.Button(btn_frame, text='Bring to Front')
    btn_back   = ttk.Button(btn_frame, text='Send to Back')
    btn_fix_zero = ttk.Button(btn_frame, text='Fix to zero')

    
    btn_save_positions = ttk.Button(btn_frame, text='Save Positions (JSON)')
    btn_load_positions = ttk.Button(btn_frame, text='Load Positions (JSON)')
    
    btn_save_values = ttk.Button(btn_frame, text='Save Tile Values (JSON)')
    btn_apply_values = ttk.Button(btn_frame, text='Load & Apply Tile Values (JSON)')

    btn_align_rows_h = ttk.Button(btn_frame, text='Align Rows (horizontal)')
    btn_align_rows_v = ttk.Button(btn_frame, text='Align Columns (vertical)')
    btn_align_diff_h = ttk.Button(btn_frame, text='Median (H)')
    btn_align_diff_v = ttk.Button(btn_frame, text='Median (V)')

    btn_level  = ttk.Button(btn_frame, text='3pt Leveling')
    btn_ptv    = ttk.Button(btn_frame, text='Peak→Valley')
    btn_save   = ttk.Button(btn_frame, text='Save Images')
    btn_mpl_view = ttk.Button(btn_frame, text='Open Matplotlib View')
    btn_save_data = ttk.Button(btn_frame, text='Save Data (GWY+(NPZ+JSON))')
    btn_quit = ttk.Button(btn_frame, text='Quit (Close without Saving)')

    
    btn_flip_h = ttk.Button(btn_frame, text='Flip Horizontal')
    btn_flip_v = ttk.Button(btn_frame, text='Flip Vertical')
    btn_rot_cw = ttk.Button(btn_frame, text='Rotate 90° CW')
    btn_rot_ccw = ttk.Button(btn_frame, text='Rotate 90° CCW')

    for b in [btn_finish, btn_save_positions, btn_load_positions, btn_front, btn_back,
             btn_fix_zero,
             btn_save_values, btn_apply_values,
             btn_align_rows_h, btn_align_rows_v,
             btn_align_diff_h, btn_align_diff_v,
             btn_level, btn_ptv, btn_flip_h, btn_flip_v, btn_rot_cw, btn_rot_ccw, btn_save, btn_mpl_view,
             btn_save_data, btn_quit]:
         b.config(width=10)
         b.pack(fill='x', padx=1, pady=1)

    
    def save_tile_value_offsets_json():
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass
        fn = os.path.join(output_dir, f"{base_filename}_tile_value_offsets.json")
        out = {}
        try:
            print(f"[debug] save_tile_value_offsets_json: output_dir={output_dir}, base_filename={base_filename}")
            print(f"[debug] ch_meta channels: {list(ch_meta.keys())}")
            for ch, meta in ch_meta.items():
                v = meta.get('value_offsets', {}) or {}
                vdisp = meta.get('value_offsets_display', {}) or {}
                
                lev = None
                try:
                    sc = ch_state.get(ch, {})
                    coeffs = sc.get('level_coeffs', None)
                    if coeffs is not None:
                        lev = [float(c) for c in coeffs]
                except Exception:
                    lev = None

                
                if v or vdisp or lev is not None:
                    entry = {}
                    if v:
                        entry['value_offsets'] = {str(k): float(vv) for k, vv in v.items()}
                    if vdisp:
                        entry['value_offsets_display'] = {str(k): float(vv) for k, vv in vdisp.items()}
                    if lev is not None:
                        entry['leveling'] = lev
                    out[ch] = entry
        except Exception as e:
            print(f"Failed to collect offsets for JSON: {e}")
            return
        if not out:
            
            data_to_write = {}
        else:
            data_to_write = out

        
        tmp_fn = fn + '.tmp'
        try:
            with open(tmp_fn, 'w', encoding='utf-8') as f:
                json.dump(data_to_write, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            
            os.replace(tmp_fn, fn)
            print(f"Saved tile value offsets to {fn} (wrote {len(data_to_write)} channel entries)")
            try:
                print(f"[debug] saved channels: {list(data_to_write.keys())}")
            except Exception:
                pass
        except Exception as e:
            try:
                
                if os.path.exists(tmp_fn): os.remove(tmp_fn)
            except Exception:
                pass
            print(f"Failed to save tile value offsets to {fn}: {e}")
            return

    def load_and_apply_tile_value_offsets_json():
        
        fn = os.path.join(output_dir, f"{base_filename}_tile_value_offsets.json")
        print(f"Loading tile value offsets from: {fn}")
        if not os.path.exists(fn):
            print(f"No offsets file found at {fn}")
            return
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read offsets JSON: {e}")
            return
        try:
            sz = os.path.getsize(fn)
            print(f"[debug] loaded file size: {sz} bytes")
        except Exception:
            pass
        try:
            print(f"Offsets JSON top-level channels: {list(data.keys())}")
        except Exception:
            pass
        applied = 0
        applied_chs = []
        skipped_chs = []
        for ch, payload in data.items():
            per_ch_applied = 0
            try:
                v = payload.get('value_offsets', {}) or {}
                lev = payload.get('leveling', None)
            except Exception:
                continue
            
            if ch in channel_matrices:
                target_ch = ch
            else:
                
                matches = [k for k in channel_matrices.keys() if k.startswith(ch)]
                target_ch = matches[0] if matches else None
            if target_ch is None:
                print(f"Channel '{ch}' from JSON not found in current session; saving offsets into ch_meta for later.")
                
                meta = ch_meta.setdefault(ch, {})
                vmap = meta.setdefault('value_offsets', {})
                for key, val in v.items():
                    try:
                        vmap[str(key)] = float(val)
                        per_ch_applied += 1
                    except Exception:
                        continue
                
                if lev is not None:
                    try:
                        meta['level_coeffs'] = [float(c) for c in lev]
                    except Exception:
                        pass
                continue

            mats = channel_matrices[target_ch]
            meta = ch_meta.setdefault(target_ch, {})
            vmap = meta.setdefault('value_offsets', {})
            
            if lev is not None:
                try:
                    ch_state.setdefault(target_ch, {})['level_coeffs'] = [float(c) for c in lev]
                except Exception:
                    pass
            for key, val in v.items():
                key_s = str(key)
                
                ks = key_s.strip()
                
                if ks.startswith('(') and ks.endswith(')'):
                    ks = ks[1:-1].strip()
                if ks.startswith('[') and ks.endswith(']'):
                    ks = ks[1:-1].strip()
                
                parts = None
                if ',' in ks:
                    parts = [p.strip() for p in ks.split(',')]
                else:
                    parts = [p for p in ks.split() if p.strip()]

                try:
                    if len(parts) >= 2:
                        r = int(parts[0]); c = int(parts[1])
                        tup = (r, c)
                    else:
                        raise ValueError('not a tile coordinate')
                except Exception:
                    
                    try:
                        vmap[ks] = float(val)
                        per_ch_applied += 1
                    except Exception:
                        
                        continue
                    continue

                
                try:
                    raw_val = float(val)
                except Exception:
                    continue

                
                
                
                key_norm = f"{tup[0]},{tup[1]}"
                prev = float(vmap.get(key_norm, 0.0))
                vmap[key_norm] = prev + raw_val
                
                try:
                    vdisp = meta.setdefault('value_offsets_display', {})
                    zm = float(meta.get('z_mult', 1.0) or 1.0)
                    prevd = float(vdisp.get(key_norm, 0.0))
                    vdisp[key_norm] = prevd + raw_val * zm
                except Exception:
                    pass
                per_ch_applied += 1

            
            applied += per_ch_applied
            if per_ch_applied > 0:
                applied_chs.append(ch)
            else:
                skipped_chs.append(ch)

        update_slider_bounds(); draw()
        print(f"Applied {applied} tile value offsets from {fn}")
        if applied_chs:
            print(f"Channels with applied entries: {applied_chs}")
        if skipped_chs:
            print(f"Channels present in JSON but no entries applied: {skipped_chs}")

    def save_data_npz_json():
        
        os.makedirs(output_dir, exist_ok=True)
        print("NPZ/JSON saving is disabled; skipping raw .npz and metadata .json output.")

        
        try:
            if gwyfile is None:
                print('gwyfile not installed; skipping .gwy output')
                return
            
            stitched_key = channels[0]
            stitched = apply_processing(build_image(channel_matrices[stitched_key]), stitched_key)
            h_px, w_px = stitched.shape[:2]

            
            meta = ch_meta.get(stitched_key, {})
            pixel_size_m = None
            try:
                if meta.get('xreal_m') is not None and meta.get('xres'):
                    pixel_size_m = float(meta['xreal_m']) / float(meta['xres'])
                elif TILE_SIZE_UM is not None:
                    
                    pixel_size_m = (float(TILE_SIZE_UM) / max(height, width)) * 1e-6
                print(f"Using pixel_size_m = {pixel_size_m} m for GWY metadata")
            except Exception:
                pixel_size_m = None

            
            try:
                from gwyfile.objects import GwyContainer, GwyDataField
            except Exception:
                GwyContainer = getattr(gwyfile, 'GwyContainer', None)
                GwyDataField = getattr(gwyfile, 'GwyDataField', None)
            if GwyContainer is None or GwyDataField is None:
                print('gwyfile objects needed to write .gwy not available; skipping .gwy')
            else:
                obj = GwyContainer()
                obj['/0/data/title'] = base_filename
                field = GwyDataField(np.asarray(stitched))
                obj['/0/data'] = field

                
                GwyInt = GwyFloat = GwySIUnit = None
                try:
                    from gwyfile.objects import GwyInt, GwyFloat, GwySIUnit
                except Exception:
                    GwyInt = GwyFloat = GwySIUnit = None

                def wrap_int(v):
                    if GwyInt is not None:
                        try:
                            return GwyInt(int(v))
                        except Exception:
                            return int(v)
                    return int(v)

                def wrap_float(v):
                    if GwyFloat is not None:
                        try:
                            return GwyFloat(float(v))
                        except Exception:
                            return float(v)
                    return float(v)

                if pixel_size_m is not None:
                    xreal = pixel_size_m * w_px
                    yreal = pixel_size_m * h_px
                    obj['/0/data/xres'] = wrap_int(w_px)
                    obj['/0/data/yres'] = wrap_int(h_px)
                    obj['/0/data/xreal'] = wrap_float(xreal)
                    obj['/0/data/yreal'] = wrap_float(yreal)

                    
                    try:
                        setattr(field, 'xres', wrap_int(w_px))
                    except Exception:
                        pass
                    try:
                        setattr(field, 'xreal', wrap_float(xreal))
                    except Exception:
                        pass

                    
                    try:
                        if GwySIUnit is not None:
                            si = GwySIUnit(unitstr='m')
                            obj['/0/data/si_unit_xy'] = si
                            try: setattr(field, 'si_unit_xy', si)
                            except Exception: pass
                    except Exception:
                        pass

                gwy_out = os.path.join(output_dir, f"{base_filename}_stitched.gwy")
                try:
                    obj.tofile(gwy_out)
                    print(f"Saved GWY stitched image to {gwy_out}")
                except Exception as e:
                    print(f"Failed to save GWY: {e}")
        except Exception as e:
            print(f"Error while writing GWY: {e}")

    
    viewer.rowconfigure(0, weight=1)
    viewer.columnconfigure(0, weight=1)

    canvas = tk.Canvas(viewer, width=900, height=800, bg='black', highlightthickness=0)
    vbar = tk.Scrollbar(viewer, orient='vertical', command=canvas.yview)
    hbar = tk.Scrollbar(viewer, orient='horizontal', command=canvas.xview)
    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

    canvas.grid(row=0, column=0, sticky='nsew')
    vbar.grid(row=0, column=1, sticky='ns')
    hbar.grid(row=1, column=0, sticky='ew')

    canvas_img = None

    
    def _parse_number_with_units(s: str):
        """Parse a numeric string with an optional unit suffix and return a float.
        Returns None if the input cannot be parsed.
        Recognized units (case-insensitive): m, cm, mm, um/µm, nm, pm and deg (degrees).
        Examples: '1.23', '1.2e-3', '50 µm', '10nm', '-5 deg'
        """
        import re
        if s is None:
            return None
        try:
            st = str(s).strip()
        except Exception:
            return None
        if st == '':
            return None
        
        st = st.replace(',', '')
        st = st.replace('\u00b5', 'u').replace('µ', 'u')
        st = st.replace('°', 'deg').replace('º', 'deg')

        
        try:
            return float(st)
        except Exception:
            pass

        m = re.match(r'^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z%]+)?\s*$', st)
        if not m:
            return None
        try:
            num = float(m.group(1))
        except Exception:
            return None
        unit = (m.group(2) or '').lower()
        
        if unit in ('',):
            return num
        if unit in ('m', 'meter', 'metre'):
            return num
        if unit in ('cm', 'centimeter', 'centimetre'):
            return num * 1e-2
        if unit in ('mm', 'millimeter', 'millimetre'):
            return num * 1e-3
        if unit in ('um', 'u', 'micron', 'micrometer', 'micrometre'):
            return num * 1e-6
        if unit in ('nm', 'nanometer', 'nanometre'):
            return num * 1e-9
        if unit in ('pm', 'picometer'):
            return num * 1e-12
        if unit in ('deg', 'degree', 'degrees'):
            
            return num
        
        return num

    def array_to_photo(arr, vmin, vmax, scale=1.0):
        arr = np.asarray(arr)
        
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
        
        cmap_name = cmap_var.get() if 'cmap_var' in locals() or 'cmap_var' in globals() else 'viridis'
        def _get_mpl_cmap(name):
            
            try:
                if name == 'gwyddion':
                    
                    return cm.get_cmap('gist_earth')
                
                return cm.get_cmap(name)
            except Exception:
                return cm.get_cmap('viridis')
        cmap_obj = _get_mpl_cmap(cmap_name)
        colored = cmap_obj(norm)
        img = (colored[..., :3] * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        try:
            if scale is not None and float(scale) != 1.0:
                nw = max(1, int(round(pil_img.width * float(scale))))
                nh = max(1, int(round(pil_img.height * float(scale))))
                pil_img = pil_img.resize((nw, nh), resample=Image.BILINEAR)
        except Exception:
            pass
        return ImageTk.PhotoImage(pil_img)

    def build_image(mats):
        logger.debug(f"build_image: building stitched image for channel '{selected}' with {len(mats)} tiles")
        big = np.zeros((path_matrix.shape[0]*height + 2*margin,
                        path_matrix.shape[1]*width  + 2*margin))
        for (r, c) in order:
            dx_off, dy_off = offsets.get((r, c), [0, 0])
            m = mats.get((r, c))
            if m is None:
                continue
            
            m = np.asarray(m)
            
            if (r, c) in base_pos:
                xs, ys = base_pos[(r, c)][0] + dx_off, base_pos[(r, c)][1] + dy_off
            else:
                ys, xs = r*height + dy_off + margin, c*width + dx_off + margin
            ye, xe = ys + height, xs + width
            
            if ys < 0 or xs < 0:
                
                
                pass
            
            if ye > big.shape[0] or xe > big.shape[1]:
                new_h = max(big.shape[0], ye + margin)
                new_w = max(big.shape[1], xe + margin)
                big_resized = np.zeros((new_h, new_w), dtype=big.dtype)
                big_resized[:big.shape[0], :big.shape[1]] = big
                big = big_resized
            
            
            targ_ys = max(ys, 0)
            targ_xs = max(xs, 0)
            targ_ye = min(ye, big.shape[0])
            targ_xe = min(xe, big.shape[1])
            rows_target = targ_ye - targ_ys
            cols_target = targ_xe - targ_xs
            if rows_target <= 0 or cols_target <= 0:
                continue
            
            src_ys = 0 if ys >= 0 else -ys
            src_xs = 0 if xs >= 0 else -xs
            rows_src = min(m.shape[0] - src_ys, rows_target)
            cols_src = min(m.shape[1] - src_xs, cols_target)
            if rows_src <= 0 or cols_src <= 0:
                continue
            
            try:
                big[targ_ys:targ_ys+rows_src, targ_xs:targ_xs+cols_src] = m[src_ys:src_ys+rows_src, src_xs:src_xs+cols_src]
            except ValueError:
                
                try:
                    big = big.astype(np.float64)
                    big[targ_ys:targ_ys+rows_src, targ_xs:targ_xs+cols_src] = m[src_ys:src_ys+rows_src, src_xs:src_xs+cols_src]
                except Exception:
                    logger.exception('Failed to blit tile into canvas')
                    continue
        return big

    def apply_processing(big, ch):
        logger.debug(f"apply_processing: applying per-channel processing for '{ch}' (state: {ch_state.get(ch)})")
        st = ch_state[ch]
        
        per_tile_processed = False
        if st.get("align_rows_h") or st.get("align_rows_v") or st.get("align_diff_h") or st.get("align_diff_v"):
            try:
                mats = channel_matrices[ch]
                proc_mats = {}
                for k, arr in mats.items():
                    a = np.asarray(arr).astype(np.float64)
                    
                    if st.get("align_rows_h"):
                        try:
                            row_med = np.median(a, axis=1, keepdims=True)
                            a = a - row_med
                        except Exception:
                            pass
                    
                    if st.get("align_rows_v"):
                        try:
                            col_med = np.median(a, axis=0, keepdims=True)
                            a = a - col_med
                        except Exception:
                            pass
                    
                    if st.get("align_diff_h"):
                        try:
                            row_med = np.median(a, axis=1, keepdims=True)
                            a = a - row_med
                        except Exception:
                            pass
                    if st.get("align_diff_v"):
                        try:
                            col_med = np.median(a, axis=0, keepdims=True)
                            a = a - col_med
                        except Exception:
                            pass
                    
                    
                    try:
                        meta = ch_meta.get(ch, {})
                        vmap = meta.get('value_offsets', {}) or {}
                        key_str = f"{k[0]},{k[1]}"
                        off_raw = float(vmap.get(key_str, 0.0)) if key_str in vmap else 0.0
                        if off_raw != 0.0:
                            a = a + off_raw
                    except Exception:
                        pass
                    proc_mats[k] = a
                
                big = build_image(proc_mats)
                per_tile_processed = True
            except Exception:
                
                per_tile_processed = False
                pass

        
        if not per_tile_processed:
            if st.get("align_diff_h") and not st.get("align_rows_h"):
                try:
                    big = big - np.median(big, axis=1, keepdims=True)
                except Exception:
                    pass
            if st.get("align_diff_v") and not st.get("align_rows_v"):
                try:
                    big = big - np.median(big, axis=0, keepdims=True)
                except Exception:
                    pass
        if st.get("level_coeffs") is not None:
            coeffs = st["level_coeffs"]
            h_, w_ = big.shape
            X, Y = np.meshgrid(np.arange(w_), np.arange(h_))
            plane = coeffs[0]*X + coeffs[1]*Y + coeffs[2]
            big = big - plane

        
        try:
            if st.get('flip_h'):
                big = np.fliplr(big)
            if st.get('flip_v'):
                big = np.flipud(big)
            rot = int(st.get('rot90', 0)) % 4
            if rot != 0:
                
                big = np.rot90(big, k=-rot)
        except Exception:
            pass
        
        
        try:
            if not per_tile_processed:
                meta = ch_meta.get(ch, {})
                vmap = meta.get('value_offsets', {}) or {}
                if vmap:
                    H, W = big.shape[:2]
                    for key_str, off_raw in list(vmap.items()):
                        try:
                            r, c = map(int, key_str.split(','))
                            dx_off, dy_off = offsets.get((r, c), (0, 0))
                            if (r, c) in base_pos:
                                xs = base_pos[(r, c)][0] + int(dx_off)
                                ys = base_pos[(r, c)][1] + int(dy_off)
                            else:
                                ys = r*height + int(dy_off) + margin
                                xs = c*width + int(dx_off) + margin
                            ye = ys + height; xe = xs + width
                            ys_i = max(0, ys); xs_i = max(0, xs)
                            ye_i = min(H, ye); xe_i = min(W, xe)
                            if ys_i < ye_i and xs_i < xe_i:
                                big[ys_i:ye_i, xs_i:xe_i] = big[ys_i:ye_i, xs_i:xe_i] + float(off_raw)
                        except Exception:
                            
                            pass
        except Exception:
            pass
        return big

    COLORBAR_WIDTH = 60; GAP = 10
    photo_cache = {'img': None, 'cbar': None}

    def update_slider_bounds():
        mats = channel_matrices[selected]
        big = build_image(mats)
        big = apply_processing(big, selected)
        
        zm = float(ch_meta.get(selected, {}).get('z_mult', 1.0))
        try:
            big = big * zm
        except Exception:
            pass
        
        vmin_new = float(np.min(big)); vmax_new = float(np.max(big))
        
        if not np.isfinite(vmin_new) or not np.isfinite(vmax_new):
            vmin_new, vmax_new = 0.0, 1.0
        span = vmax_new - vmin_new
        if span <= 0 or not np.isfinite(span):
            
            
            mag = max(abs(vmin_new), abs(vmax_new), 1.0)
            span = mag * 1e-6
            vmax_new = vmin_new + span
        
        
        resolution = max(span / 10000.0, 1e-12)

        
        
        try:
            cur_vmin = float(vmin_scale.get())
            cur_vmax = float(vmax_scale.get())
        except Exception:
            cur_vmin, cur_vmax = vmin_new, vmax_new

        
        if cur_vmax <= cur_vmin:
            cur_vmin, cur_vmax = vmin_new, vmax_new

        
        cur_vmin = max(vmin_new, min(cur_vmin, vmax_new))
        cur_vmax = max(vmin_new, min(cur_vmax, vmax_new))

        
        vmin_scale.config(from_=vmin_new, to=vmax_new, resolution=resolution)
        vmax_scale.config(from_=vmin_new, to=vmax_new, resolution=resolution)
        vmin_scale.set(cur_vmin)
        vmax_scale.set(cur_vmax)
        
        try:
            H, W = big.shape[:2]
            crop_top.config(to=max(0, H-1))
            crop_bottom.config(to=max(0, H-1))
            crop_left.config(to=max(0, W-1))
            crop_right.config(to=(max(0, W-1)))
            
            if crop_top_var.get() + crop_bottom_var.get() >= H:
                crop_bottom_var.set(max(0, H-1 - crop_top_var.get()))
            if crop_left_var.get() + crop_right_var.get() >= W:
                crop_right_var.set(max(0, W-1 - crop_left_var.get()))
        except Exception:
            pass

    def draw():
        nonlocal canvas_img, cur_img_w, cur_img_h
        logger.debug(f"draw: redrawing canvas for channel '{selected}'")
        mats = channel_matrices[selected]
        big = build_image(mats)
        big = apply_processing(big, selected)
        
        zm = float(ch_meta.get(selected, {}).get('z_mult', 1.0))
        try:
            big = big * zm
        except Exception:
            pass
        
        try:
            lo, hi = float(vmin_scale.get()), float(vmax_scale.get())
        except Exception:
            lo, hi = auto_minmax(big)
        
        try:
            H, W = big.shape[:2]
            ct = int(max(0, min(crop_top_var.get(), H-1)))
            cb = int(max(0, min(crop_bottom_var.get(), H-1)))
            cl = int(max(0, min(crop_left_var.get(), W-1)))
            cr = int(max(0, min(crop_right_var.get(), W-1)))
            y0 = ct; y1 = max(y0+1, H - cb)
            x0 = cl; x1 = max(x0+1, W - cr)
            big_disp = big[y0:y1, x0:x1]
        except Exception:
            big_disp = big

        
        pil_img = array_to_photo(big_disp, lo, hi, scale=zoom)
        canvas_img = pil_img
        canvas.delete("all")
        img_w = pil_img.width(); img_h = pil_img.height()
        
        cur_img_w = img_w; cur_img_h = img_h
        canvas.create_image(0, 0, anchor='nw', image=canvas_img)
        
        COLORBAR_WIDTH = 60; GAP = 10
        try:
            
            grad = np.linspace(hi, lo, img_h, dtype=float).reshape(img_h, 1)
            grad = np.repeat(grad, COLORBAR_WIDTH, axis=1)
            cbar_photo = array_to_photo(grad, lo, hi, scale=1.0)
            photo_cache['cbar'] = cbar_photo
            x_cbar = img_w + GAP
            canvas.create_image(x_cbar, 0, anchor='nw', image=cbar_photo)
            canvas.create_rectangle(x_cbar-1, -1, x_cbar+COLORBAR_WIDTH+1, img_h+1, outline='white')
        except Exception:
            pass

        
        
        unit = channel_unit(selected)
        def fmt(v):
            if v == 0: return '0'
            mag = abs(v)
            if mag >= 1e3 or mag < 1e-2: return f"{v:.3e}"
            if mag < 1: return f"{v:.3f}"
            return f"{v:.2f}"
        y_top, y_mid, y_bot = 0, img_h/2, img_h
        med_val = float(np.median(big_disp))
        canvas.create_text(x_cbar+COLORBAR_WIDTH+4, y_top+8, anchor='nw', fill='white', text=f"{fmt(hi)} {unit}")
        canvas.create_text(x_cbar+COLORBAR_WIDTH+4, y_mid-7, anchor='nw', fill='white', text=f"{fmt(med_val)}")
        canvas.create_text(x_cbar+COLORBAR_WIDTH+4, y_bot-16, anchor='nw', fill='white', text=f"{fmt(lo)}")
        total_w = img_w + GAP + COLORBAR_WIDTH 
        total_w = max(total_w, canvas.winfo_width())
        total_h = img_h + GAP
        total_h = max(total_h, canvas.winfo_height())
        canvas.config(scrollregion=(0, 0, total_w, total_h))
        
        if active is not None:
            r, c = active
            dx, dy = offsets[active]
            if (r, c) in base_pos:
                x0 = base_pos[(r, c)][0] + dx
                y0 = base_pos[(r, c)][1] + dy
            else:
                x0, y0 = c*width + dx + margin, r*height + dy + margin
            x1 = x0 + width
            y1 = y0 + height
            
            try:
                cl = int(crop_left_var.get()); ct = int(crop_top_var.get())
                x0d = x0 - cl; y0d = y0 - ct; x1d = x1 - cl; y1d = y1 - ct
            except Exception:
                x0d, y0d, x1d, y1d = x0, y0, x1, y1
            
            try:
                if zoom != 1.0:
                    x0d = x0d * zoom; y0d = y0d * zoom; x1d = x1d * zoom; y1d = y1d * zoom
            except Exception:
                pass
            canvas.create_rectangle(x0d, y0d, x1d, y1d, outline='red', width=2)

        
        try:
            if scale_enabled_var.get():
                meta = ch_meta.get(selected, {})
                px_um = None
                try:
                    if meta.get('xreal_m') is not None and meta.get('xres'):
                        px_um = float(meta['xreal_m']) / float(meta['xres']) * 1e6
                    else:
                        px_um = meta.get('pixel_size_x_um', None) or meta.get('pixel_size_um', None)
                except Exception:
                    px_um = None
                label = scale_label_var.get()
                try:
                    sw = float(scale_width_var.get())
                except Exception:
                    sw = None
                if px_um is not None and sw is not None:
                    pixel_len = int(round(sw / float(px_um) * zoom))
                    
                    x = 10
                    y = img_h - 20
                    canvas.create_rectangle(x, y, x + pixel_len, y + 6, fill='white', outline='black')
                    canvas.create_text(x + pixel_len/2, y + 6 + 10, text=label, fill='white')
                else:
                    
                    canvas.create_text(20, img_h - 10, anchor='w', text=label, fill='white')
        except Exception:
            pass

    
    def ensure_canvas_focus(_evt=None):
        try: canvas.focus_set()
        except Exception: pass
    ensure_canvas_focus()

    def nudge_active(dx: int, dy: int, step: int = 1):
        nonlocal active
        if active is None: return
        off = offsets[active]; off[0] += int(dx)*int(step); off[1] += int(dy)*int(step)
        offsets[active] = off; draw()

    viewer.bind("<Enter>", ensure_canvas_focus)
    canvas.bind("<Enter>", ensure_canvas_focus)
    canvas.bind("<Up>",   lambda e: nudge_active(0, -1, 1))
    canvas.bind("<Down>", lambda e: nudge_active(0,  1, 1))
    canvas.bind("<Left>", lambda e: nudge_active(-1, 0, 1))
    canvas.bind("<Right>",lambda e: nudge_active( 1, 0, 1))
    canvas.bind("<Shift-Up>",   lambda e: nudge_active(0, -1, 10))
    canvas.bind("<Shift-Down>", lambda e: nudge_active(0,  1, 10))
    canvas.bind("<Shift-Left>", lambda e: nudge_active(-1, 0, 10))
    canvas.bind("<Shift-Right>",lambda e: nudge_active( 1, 0, 10))
    canvas.bind("<Alt-Up>",   lambda e: nudge_active(0, -1, 50))
    canvas.bind("<Alt-Down>", lambda e: nudge_active(0,  1, 50))
    canvas.bind("<Alt-Left>", lambda e: nudge_active(-1, 0, 50))
    canvas.bind("<Alt-Right>",lambda e: nudge_active( 1, 0, 50))

    
    def on_w_key(_event=None):
        if tile_value_edit_mode_var.get():
            inc_tile_value()
        else:
            nudge_tile_up()
    def on_s_key(_event=None):
        if tile_value_edit_mode_var.get():
            dec_tile_value()
        else:
            nudge_tile_down()

    canvas.bind("w", lambda e: on_w_key(e))
    canvas.bind("s", lambda e: on_s_key(e))
    canvas.bind("W", lambda e: on_w_key(e))
    canvas.bind("S", lambda e: on_s_key(e))

    
    def finish():
        fn = os.path.join(output_dir, f"{base_filename}_offsets.txt")
        with open(fn, 'w') as f:
            for (r, c), (dx, dy) in offsets.items(): f.write(f"{r} {c} {dx} {dy}\n")
        logger.info(f"Offsets written to {fn}")
        print(f"Offsets -> {fn}")
        viewer.destroy(); controls.destroy(); root.destroy()

    def load_offsets():
        fn = os.path.join(output_dir, f"{base_filename}_offsets.txt")
        if not os.path.exists(fn):
            logger.warning('No offsets file found to load')
            print("No offsets file."); return
        with open(fn) as f:
            for ln in f:
                r, c, dx, dy = map(int, ln.split())
                offsets[(r, c)] = [dx, dy]
        logger.info(f"Loaded offsets from {fn}: {len(offsets)} entries")
        draw()

    def save_positions_json():
        """Save current positions (offsets, base_pos, order) to JSON in output_dir."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            fn = os.path.join(output_dir, f"{base_filename}_positions.json")
            data = {
                'offsets': {f"{r},{c}": [int(dx), int(dy)] for (r,c), (dx,dy) in offsets.items()},
                'base_pos': {f"{r},{c}": [int(v[0]), int(v[1])] for (r,c), v in base_pos.items()},
                'order': [f"{r},{c}" for (r,c) in order]
            }
            with open(fn, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved positions JSON to {fn}")
            print(f"Saved positions -> {fn}")
        except Exception as e:
            print(f"Failed to save positions JSON: {e}")

    def load_positions_json():
        """Load positions JSON and apply to current viewer state (offsets, base_pos, order)."""
        fn = os.path.join(output_dir, f"{base_filename}_positions.json")
        if not os.path.exists(fn):
            logger.warning('No positions JSON found to load')
            print("No positions JSON file."); return
        try:
            with open(fn, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read positions JSON: {e}"); return
        try:
            offs = data.get('offsets', {})
            for k, v in offs.items():
                r, c = map(int, k.split(','))
                offsets[(r, c)] = [int(v[0]), int(v[1])]
            bp = data.get('base_pos', {})
            base_pos.clear()
            for k, v in bp.items():
                r, c = map(int, k.split(','))
                base_pos[(r, c)] = [int(v[0]), int(v[1])]
            ord_list = data.get('order')
            if ord_list:
                order.clear()
                for s in ord_list:
                    r, c = map(int, s.split(','))
                    order.append((r, c))
            logger.info(f"Loaded positions JSON from {fn}")
            print(f"Loaded positions <- {fn}")
            draw()
        except Exception as e:
            print(f"Failed to apply positions JSON: {e}")

    
    btn_save_positions.configure(command=save_positions_json)
    btn_load_positions.configure(command=load_positions_json)

    def bring_to_front():
        nonlocal order
        if active in order:
            order.remove(active); order.append(active); draw()

    def send_to_back():
        nonlocal order
        if active in order:
            order.remove(active); order.insert(0, active); draw()

    def on_channel_change(*_):
        nonlocal selected, active
        
        if channels:
            selected = channels[0]
        active = None; mode.set("none")
        compute_base_positions_for_channel(selected)
        update_slider_bounds();
        try:
            update_tile_dropdown()
        except Exception:
            pass
        draw()
        update_align_button_states()

    def update_align_button_states():
        st = ch_state[selected]
        btn_align_rows_h.config(text=('✓ ' if st['align_rows_h'] else '') + 'Align Rows (horizontal)')
        btn_align_rows_v.config(text=('✓ ' if st['align_rows_v'] else '') + 'Align Columns (vertical)')
        btn_align_diff_h.config(text=('✓ ' if st['align_diff_h'] else '') + 'Median (H)')
        btn_align_diff_v.config(text=('✓ ' if st['align_diff_v'] else '') + 'Median (V)')
        
        btn_flip_h.config(text=(('✓ ' if st.get('flip_h') else '') + 'Flip Horizontal'))
        btn_flip_v.config(text=(('✓ ' if st.get('flip_v') else '') + 'Flip Vertical'))
        angle = (int(st.get('rot90', 0)) * 90) % 360
        btn_rot_cw.config(text=(f'Rotate 90° CW (cur: {angle}°)'))
        btn_rot_ccw.config(text=(f'Rotate 90° CCW (cur: {angle}°)'))

    def toggle_align_rows_h():
        st = ch_state[selected]; st["align_rows_h"] = not st["align_rows_h"]
        update_slider_bounds(); draw(); update_align_button_states()

    def toggle_align_rows_v():
        st = ch_state[selected]; st["align_rows_v"] = not st["align_rows_v"]
        update_slider_bounds(); draw(); update_align_button_states()

    def toggle_align_diff_h():
        st = ch_state[selected]; st["align_diff_h"] = not st["align_diff_h"]
        update_slider_bounds(); draw(); update_align_button_states()

    def toggle_align_diff_v():
        st = ch_state[selected]; st["align_diff_v"] = not st["align_diff_v"]
        update_slider_bounds(); draw(); update_align_button_states()

    
    def toggle_flip_h():
        st = ch_state[selected]
        st['flip_h'] = not bool(st.get('flip_h'))
        draw(); update_align_button_states()

    def toggle_flip_v():
        st = ch_state[selected]
        st['flip_v'] = not bool(st.get('flip_v'))
        draw(); update_align_button_states()

    def rotate_cw():
        st = ch_state[selected]
        st['rot90'] = (int(st.get('rot90', 0)) + 1) % 4
        draw(); update_align_button_states()

    def rotate_ccw():
        st = ch_state[selected]
        st['rot90'] = (int(st.get('rot90', 0)) - 1) % 4
        draw(); update_align_button_states()

    def start_leveling():
        ch_state[selected]["level_points"] = []
        ch_state[selected]["level_coeffs"] = None
        mode.set("leveling")
        print("3pt leveling: click 3 points on the viewer.")

    ptv_points = []
    def start_ptv():
        ptv_points.clear(); mode.set("ptv"); print("Peak→Valley: click two points on the viewer.")


    def save_images():
        vmin, vmax = float(vmin_scale.get()), float(vmax_scale.get())
        extent = None
        for ch in channels:
            big = apply_processing(build_image(channel_matrices[ch]), ch)
            
            try:
                H, W = big.shape[:2]
                ct = int(max(0, min(crop_top_var.get(), H-1)))
                cb = int(max(0, min(crop_bottom_var.get(), H-1)))
                cl = int(max(0, min(crop_left_var.get(), W-1)))
                cr = int(max(0, min(crop_right_var.get(), W-1)))
                y0 = ct; y1 = max(y0+1, H - cb)
                x0 = cl; x1 = max(x0+1, W - cr)
                img = big[y0:y1, x0:x1]
            except Exception:
                img = big
            
            zm = float(ch_meta.get(ch, {}).get('z_mult', 1.0))
            try:
                img = img * zm
            except Exception:
                pass
            img = np.clip(img, vmin, vmax)
            
            meta = ch_meta.get(ch, {})
            px_um = None
            try:
                if meta.get('xreal_m') is not None and meta.get('xres'):
                    px_um = float(meta['xreal_m']) / float(meta['xres']) * 1e6
                elif meta.get('pixel_size_um') not in (None, ''):
                    px_um = float(meta.get('pixel_size_um'))
            except Exception:
                px_um = None

            out_path = os.path.join(output_dir, f"{base_filename}_{ch}_final.png")
            
            
            h_px, w_px = img.shape[:2]
            dpi = 100
            fig_w = max(2, min(20, w_px / dpi))
            fig_h = max(2, min(20, h_px / dpi))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            
            cmap_obj = None
            try:
                cmap_name = cmap_var.get()
                cmap_obj = cm.get_cmap('gist_earth') if cmap_name == 'gwyddion' else cm.get_cmap(cmap_name)
            except Exception:
                cmap_obj = cm.get_cmap('viridis')
            
            extent = None
            try:
                if px_um is not None:
                    h_img, w_img = img.shape[:2]
                    extent = [0, w_img * float(px_um), 0, h_img * float(px_um)]
            except Exception:
                extent = None
            try:
                if extent is not None:
                    im = ax.imshow(img, cmap=cmap_obj, origin='lower', vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
                else:
                    im = ax.imshow(img, cmap=cmap_obj, origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
            except Exception:
                im = ax.imshow(img, cmap=cmap_obj, origin='lower', vmin=vmin, vmax=vmax, aspect='equal')

            
            try:
                if scale_enabled_var.get():
                    sw = float(scale_width_var.get())
                    label = scale_label_var.get()
                    if px_um is not None and extent is not None:
                        
                        x0 = 0.05 * (extent[1] - extent[0])
                        y0 = 0.05 * (extent[3] - extent[2])
                        bar_h = 0.02 * (extent[3] - extent[2])
                        from matplotlib.patches import Rectangle
                        rect = Rectangle((x0, y0), sw, bar_h, facecolor='white', edgecolor='black')
                        ax.add_patch(rect)
                        ax.text(x0 + sw/2, y0 + bar_h + 0.01*(extent[3]-extent[2]), label,
                                ha='center', va='bottom', color='white', fontsize=8)
                    else:
                        
                        ax.text(0.05, 0.05, label, transform=ax.transAxes, color='white', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.5))
            except Exception:
                pass

            ax.axis('off')
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        print(f"Saved outputs to {output_dir}")

    def open_matplotlib_view():
        extent = None
        mats = channel_matrices[selected]
        big = apply_processing(build_image(mats), selected)
        vmin, vmax = float(vmin_scale.get()), float(vmax_scale.get())

        
        try:
            H, W = big.shape[:2]
            ct = int(max(0, min(crop_top_var.get(), H-1)))
            cb = int(max(0, min(crop_bottom_var.get(), H-1)))
            cl = int(max(0, min(crop_left_var.get(), W-1)))
            cr = int(max(0, min(crop_right_var.get(), W-1)))
            y0 = ct; y1 = max(y0+1, H - cb)
            x0 = cl; x1 = max(x0+1, W - cr)
            img = big[y0:y1, x0:x1]
        except Exception:
            img = big

        
        zm = float(ch_meta.get(selected, {}).get('z_mult', 1.0))
        try:
            img = img * zm
        except Exception:
            pass

        
        px_um = None
        try:
            meta = ch_meta.get(selected, {})
            
            if meta.get('pixel_size_um') not in (None, ''):
                px_um = float(meta.get('pixel_size_um'))
            
            elif meta.get('pixel_size_m') not in (None, ''):
                px_um = float(meta.get('pixel_size_m')) * 1e6
            
            elif meta.get('xreal_m') not in (None, '') and meta.get('xres'):
                px_um = float(meta.get('xreal_m')) / float(meta.get('xres')) * 1e6
            
            elif meta.get('tile_size_um_x') not in (None, '') and meta.get('xres'):
                px_um = float(meta.get('tile_size_um_x')) / float(meta.get('xres'))
            
            elif meta.get('tile_size_um_y') not in (None, '') and meta.get('yres'):
                px_um = float(meta.get('tile_size_um_y')) / float(meta.get('yres'))
        except Exception:
            px_um = None

        
        if px_um is None:
            try:
                h_px, w_px = img.shape[:2]
                if TILE_SIZE_UM is not None and max(h_px, w_px) > 0:
                    px_um = float(TILE_SIZE_UM) / float(max(h_px, w_px))
            except Exception:
                px_um = None

        
        if px_um is None:
            print(f"[info] Matplotlib view: no pixel-size metadata for channel '{selected}'. Scale/µm axes unavailable.")
        else:
            print(f"[info] Matplotlib view: using pixel size {px_um:.6g} µm/px for channel '{selected}'")

        img_to_show = np.clip(img, vmin, vmax)

        fig, ax = plt.subplots()
        extent = None
        if px_um is not None:
            h, w = img_to_show.shape[:2]
            width_um = w * float(px_um)
            height_um = h * float(px_um)
            extent = [0, width_um, 0, height_um]
            
            try:
                cmap_name = cmap_var.get()
                cmap_obj = cm.get_cmap('gist_earth') if cmap_name == 'gwyddion' else cm.get_cmap(cmap_name)
            except Exception:
                cmap_obj = cm.get_cmap('viridis')
            im = ax.imshow(img_to_show, cmap=cmap_obj, origin='lower', vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
            ax.set_xlabel('X (µm)')
            ax.set_ylabel('Y (µm)')
        else:
            
            h, w = img_to_show.shape[:2]
            extent = [0, w, 0, h]
            try:
                cmap_name = cmap_var.get()
                cmap_obj = cm.get_cmap('gist_earth') if cmap_name == 'gwyddion' else cm.get_cmap(cmap_name)
            except Exception:
                cmap_obj = cm.get_cmap('viridis')
            im = ax.imshow(img_to_show, cmap=cmap_obj, origin='lower', vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')

        
        try:
            if scale_enabled_var.get():
                
                try:
                    sw_requested = float(scale_width_var.get())
                except Exception:
                    sw_requested = None
                label = scale_label_var.get()
                if px_um is not None and sw_requested is not None:
                    
                    x0_bar = extent[0] + 0.05 * (extent[1] - extent[0])
                    y0_bar = extent[2] + 0.05 * (extent[3] - extent[2])
                    bar_h = 0.02 * (extent[3] - extent[2])
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x0_bar, y0_bar), sw_requested, bar_h, facecolor='white', edgecolor='black')
                    ax.add_patch(rect)
                    ax.text(x0_bar + sw_requested/2, y0_bar + bar_h + 0.01*(extent[3]-extent[2]), label,
                            ha='center', va='bottom', color='white', fontsize=8)
                elif sw_requested is not None:
                    
                    try:
                        sw_px = int(round(sw_requested))
                    except Exception:
                        sw_px = min(100, w//10)
                    x0_bar = extent[0] + 0.05 * (extent[1] - extent[0])
                    y0_bar = extent[2] + 0.05 * (extent[3] - extent[2])
                    bar_h = 0.02 * (extent[3] - extent[2])
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x0_bar, y0_bar), sw_px, bar_h, facecolor='white', edgecolor='black')
                    ax.add_patch(rect)
                    ax.text(x0_bar + sw_px/2, y0_bar + bar_h + 0.01*(extent[3]-extent[2]), label + ' (px)',
                            ha='center', va='bottom', color='white', fontsize=8)
        except Exception:
            pass

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(channel_unit(selected))
        plt.title(f"Matplotlib View — {selected}")
        plt.tight_layout()
        try:
            plt.show(block=False)
        except TypeError:
            plt.show()

    def fix_to_zero():
        """Shift all tile arrays in the selected channel so that the global minimum becomes zero.
        This modifies the data in channel_matrices in-place (arrays are converted to float32).
        Records the applied offset in channel metadata under keys fixed_zero*, and updates the view.
        """
        mats = channel_matrices[selected]
        if not mats:
            print(f"No tiles for channel {selected}")
            return
        
        gmin = None
        for k, arr in mats.items():
            try:
                a = np.asarray(arr)
                mv = float(np.nanmin(a))
            except Exception:
                continue
            if gmin is None or mv < gmin:
                gmin = mv
        if gmin is None:
            print("Could not determine global minimum; aborting fix-to-zero")
            return
        offset = -float(gmin)
        if abs(offset) < 1e-12:
            print(f"Channel {selected} already has minimum >= 0 (min={gmin})")
            
            ch_meta.setdefault(selected, {})['fixed_zero'] = False
            ch_meta[selected]['fixed_zero_original_min'] = float(gmin)
            ch_meta[selected]['fixed_zero_offset'] = 0.0
            return
        
        for k in list(mats.keys()):
            arr = np.asarray(mats[k]).astype(np.float64)
            mats[k] = (arr + offset)
        channel_matrices[selected] = mats
        ch_meta.setdefault(selected, {})['fixed_zero'] = True
        ch_meta[selected]['fixed_zero_offset'] = float(offset)
        ch_meta[selected]['fixed_zero_original_min'] = float(gmin)
        update_slider_bounds(); draw()
        print(f"Applied Fix-to-zero to channel '{selected}': added {offset} (original min {gmin})")

    def quit_all():
        try:
            viewer.destroy()
        except Exception:
            pass
        try:
            controls.destroy()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)
        
    btn_finish.configure(command=finish)
    
    
    btn_save_values.configure(command=save_tile_value_offsets_json)
    btn_apply_values.configure(command=load_and_apply_tile_value_offsets_json)
    btn_front.configure(command=bring_to_front)
    btn_back.configure(command=send_to_back)
    btn_fix_zero.configure(command=fix_to_zero)
    btn_align_rows_h.configure(command=toggle_align_rows_h)
    btn_align_rows_v.configure(command=toggle_align_rows_v)
    btn_align_diff_h.configure(command=toggle_align_diff_h)
    btn_align_diff_v.configure(command=toggle_align_diff_v)
    btn_level.configure(command=start_leveling)
    btn_ptv.configure(command=start_ptv)
    btn_save.configure(command=save_images)
    btn_mpl_view.configure(command=open_matplotlib_view)
    btn_save_data.configure(command=save_data_npz_json)
    btn_quit.configure(command=quit_all)
    
    btn_flip_h.configure(command=toggle_flip_h)
    btn_flip_v.configure(command=toggle_flip_v)
    btn_rot_cw.configure(command=rotate_cw)
    btn_rot_ccw.configure(command=rotate_ccw)
    
    canvas.bind('h', lambda e: toggle_align_rows_h())
    canvas.bind('v', lambda e: toggle_align_rows_v())
    canvas.bind('m', lambda e: toggle_align_diff_h())
    canvas.bind('n', lambda e: toggle_align_diff_v())
    
    canvas.bind('H', lambda e: toggle_align_rows_h())
    canvas.bind('V', lambda e: toggle_align_rows_v())
    canvas.bind('M', lambda e: toggle_align_diff_h())
    canvas.bind('N', lambda e: toggle_align_diff_v())
    
    canvas.bind('f', lambda e: toggle_flip_h())
    canvas.bind('F', lambda e: toggle_flip_v())
    canvas.bind('r', lambda e: rotate_cw())
    canvas.bind('R', lambda e: rotate_ccw())
    
    def on_vmin_change(_=None):
        try:
            vmin = float(vmin_scale.get()); vmax = float(vmax_scale.get())
            if vmin >= vmax: vmin_scale.set(vmax - 1e-6)
        except Exception: pass
        draw()
    def on_vmax_change(_=None):
        try:
            vmin = float(vmin_scale.get()); vmax = float(vmax_scale.get())
            if vmax <= vmin: vmax_scale.set(vmin + 1e-6)
        except Exception: pass
        draw()
    vmin_scale.configure(command=lambda _val: on_vmin_change())
    vmax_scale.configure(command=lambda _val: on_vmax_change())   
    
    def canvas_click(event):
        nonlocal active
        logger.debug(f"canvas_click: x={event.x} y={event.y} mode={mode.get()} active_before={active}")
        x, y = event.x, event.y
        current_mode = mode.get()
        if current_mode == "leveling":
            pts = ch_state[selected]["level_points"]
            pts.append((x, y))
            if len(pts) == 3:
                mats = channel_matrices[selected]
                big = apply_processing(build_image(mats), selected)
                A = []; bvals = []
                for (px, py) in pts:
                    xi = max(0, min(big.shape[1]-1, int(round(px))))
                    yi = max(0, min(big.shape[0]-1, int(round(py))))
                    A.append([xi, yi, 1.0]); bvals.append(big[yi, xi])
                A = np.array(A, float); bvec = np.array(bvals, float)
                try:
                    coeffs, *_ = np.linalg.lstsq(A, bvec, rcond=None)
                    ch_state[selected]["level_coeffs"] = coeffs
                    print(f"Leveling plane set for {selected}: {coeffs}")
                except Exception as e:
                    print(f"Leveling failed: {e}")
                finally:
                    mode.set("none")
                update_slider_bounds(); draw()
            else:
                draw()
            return
        elif current_mode == "ptv":
            ptv_points.append((x, y))
            if len(ptv_points) == 2:
                mats = channel_matrices[selected]
                big = apply_processing(build_image(mats), selected)
                x0, y0 = ptv_points[0]; x1, y1 = ptv_points[1]
                length = int(max(1, np.hypot(x1 - x0, y1 - y0)))
                xs = np.linspace(x0, x1, length); ys = np.linspace(y0, y1, length)
                profile = map_coordinates(big, np.vstack([ys, xs]), order=1)
                import matplotlib.figure as mplfig
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                fig = mplfig.Figure(figsize=(5, 3), dpi=100)
                axp = fig.add_subplot(111); axp.plot(np.arange(len(profile)), profile)
                axp.set_title(f"Peak → Valley: {selected}"); axp.set_xlabel("Distance (px)"); axp.set_ylabel("Value"); axp.grid(True)
                win = tk.Toplevel(viewer); win.title("Peak→Valley profile")
                canvas_fig = FigureCanvasTkAgg(fig, master=win); canvas_fig.draw(); canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                ptv_points.clear(); mode.set("none")
            draw(); return
        
        mats = channel_matrices[selected]
        
        try:
            cl = int(crop_left_var.get()); ct = int(crop_top_var.get())
        except Exception:
            cl = 0; ct = 0
        x_glob = x + cl; y_glob = y + ct

        for (r, c), (dx, dy) in offsets.items():
            if (r, c) not in mats: continue
            if (r, c) in base_pos:
                x0 = base_pos[(r, c)][0] + dx
                y0 = base_pos[(r, c)][1] + dy
            else:
                x0, y0 = c*width + dx + margin, r*height + dy + margin
            if x0 <= x_glob <= x0+width and y0 <= y_glob <= y0+height:
                active = (r, c); break
        draw()

    canvas.bind("<Button-1>", canvas_click)

    
    def _on_mousewheel(event):
        delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, 'units')
    def _on_shift_mousewheel(event):
        delta = -1 if event.delta > 0 else 1
        canvas.xview_scroll(delta, 'units')
    canvas.bind_all('<MouseWheel>', _on_mousewheel)
    canvas.bind_all('<Shift-MouseWheel>', _on_shift_mousewheel)

    
    def on_zoom(event):
        nonlocal zoom, cur_img_w, cur_img_h
        try:
            delta = event.delta
        except Exception:
            
            delta = 0
            try:
                if event.num == 4: delta = 120
                elif event.num == 5: delta = -120
            except Exception:
                pass
        if delta == 0:
            return
        
        factor = zoom_step if delta > 0 else (1.0/zoom_step)
        new_zoom = max(min_zoom, min(max_zoom, zoom * factor))
        if abs(new_zoom - zoom) < 1e-6:
            return
        
        try:
            ix = canvas.canvasx(event.x); iy = canvas.canvasy(event.y)
            fx = float(ix) / float(cur_img_w) if cur_img_w else 0.5
            fy = float(iy) / float(cur_img_h) if cur_img_h else 0.5
        except Exception:
            fx = 0.5; fy = 0.5
        zoom = new_zoom
        
        draw()
        
        try:
            new_w = cur_img_w; new_h = cur_img_h
            total_w = canvas.bbox('all')[2] if canvas.bbox('all') else max(new_w + GAP + COLORBAR_WIDTH, canvas.winfo_width())
            total_h = canvas.bbox('all')[3] if canvas.bbox('all') else max(new_h + GAP, canvas.winfo_height())
            left = fx * new_w - event.x
            top = fy * new_h - event.y
            canvas.xview_moveto(max(0.0, min(1.0, left / float(max(1, total_w)))))
            canvas.yview_moveto(max(0.0, min(1.0, top / float(max(1, total_h)))))
        except Exception:
            pass

    
    try:
        canvas.bind_all('<Control-MouseWheel>', on_zoom)
    except Exception:
        pass
    try:
        canvas.bind_all('<Command-MouseWheel>', on_zoom)
    except Exception:
        pass

    
    compute_base_positions_for_channel(selected)
    update_align_button_states()
    update_slider_bounds();
    try:
        update_tile_dropdown()
    except Exception:
        pass
    draw(); root.mainloop()


def main():
    setup_logging()
    
    def choose_folder_dialog():
        
        
        path = None
        if sys.platform == 'darwin':
            try:
                cmd = ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select Tiles Folder")']
                res = subprocess.run(cmd, capture_output=True, text=True)
                p = res.stdout.strip()
                if p:
                    return p
            except Exception:
                pass
        if sys.platform.startswith('linux'):
            try:
                import shutil
                if shutil.which('zenity'):
                    cmd = ['zenity', '--file-selection', '--directory', '--title=Select Tiles Folder']
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    p = res.stdout.strip()
                    if p:
                        return p
            except Exception:
                pass
        
        try:
            root = tk.Tk(); root.withdraw()
            p = filedialog.askdirectory(title='Select Tiles Folder')
            try: root.destroy()
            except Exception: pass
            if p:
                return p
        except Exception:
            pass
        return None

    chosen = choose_folder_dialog()
    if not chosen:
        print('No folder chosen; exiting')
        sys.exit(0)
    cmats, P, h, w, tile_names, sizes, all_files, ch_meta = load_gwy_tiles(chosen, CENTER_CROP)
    
    interactive_tiff_view(cmats, P, h, w, chosen, BASE_FILENAME, channel_metadata=ch_meta, tile_names=tile_names)
    
    
if __name__ == '__main__':
    main()