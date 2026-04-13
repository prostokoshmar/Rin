#from scikit_ransac_afim import main as find_main
from cv2_ransac import main as find_main
import argparse

def _str_to_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() in ("1", "true", "yes", "y", "on")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ransac_main with specified parameters.")
    parser.add_argument('--methods', type=lambda s: [m.strip() for m in s.split(',') if m.strip()], required=True,
                        help="Comma-separated list of methods (e.g. sift,orb)")
    parser.add_argument('--tile_dir', type=str, required=False, default=None, help="Path to tiles folder")
    parser.add_argument('--matching_ratio', type=float, required=False, default=None, help="matching_ratio_param (float)")
    parser.add_argument('--matching_angle', type=float, required=False, default=None, help="matching_angle_param (float)")
    parser.add_argument('--min_matches', type=int, required=False, default=None, help="min_matches_param (int)")
    parser.add_argument('--len_gap', type=int, required=False, default=None, help="len_gap_param (int)")
    parser.add_argument('--selected_method', type=str, required=False, default=None, help="selected_method_param (str)")
    parser.add_argument('--three_point_level', action='store_true', help="Enable per-tile 3-point leveling (three_point_level_param)")
    parser.add_argument('--ransac_enabled', action='store_true', help="Enable RANSAC (ransac_enabled_param)")
    parser.add_argument('--ransac_thresh', type=float, required=False, default=5.0, help="ransac_thresh_param (float)")
    parser.add_argument('--ransac_min_inliers', type=int, required=False, default=4, help="ransac_min_inliers_param (int)")
    parser.add_argument('--preprocess_enabled', type=_str_to_bool, required=False, default=True, help="preprocess_enabled_param (bool) - use true/false")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    find_main(
        methods=args.methods,
        tile_dir=args.tile_dir,
        matching_ratio_param=args.matching_ratio,
        matching_angle_param=args.matching_angle,
        min_matches_param=args.min_matches,
        len_gap_param=args.len_gap,
        selected_method_param=args.selected_method,
        three_point_level_param=bool(args.three_point_level),
        ransac_enabled_param=bool(args.ransac_enabled),
        ransac_thresh_param=args.ransac_thresh,
        ransac_min_inliers_param=args.ransac_min_inliers,
        preprocess_enabled_param=bool(args.preprocess_enabled)
    )
    
    #Terminal command example:
    """python3 agparse.py --methods sift --tile_dir "/Volumes/T7/last/NEWDATA/60V/MFMPhase_Backward" --matching_ratio 0.1 --matching_angle 10 --min_matches 4 --len_gap 20 --selected_method sift --ransac_enabled --three_point_level --ransac_thresh 5.0 --ransac_min_inliers 4 --preprocess_enabled true"""