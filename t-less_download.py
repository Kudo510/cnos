# Authors: Tomas Hodan (hodantom@cmp.felk.cvut.cz), Pavel Haluza
# Center for Machine Perception, Czech Technical University in Prague

import sys
import shutil
import os
import tempfile
import math
import zipfile
import argparse
from typing import List, Optional, Union
from urllib.parse import urlparse, quote, urlunsplit, urlsplit

# Constants
VERSION = 2
URL_ROOT = f"http://ptak.felk.cvut.cz/darwin/t-less/v{VERSION}"
TRAIN_IDS = list(range(1, 31))
TEST_IDS = list(range(1, 21))
T_TYPES = ["train", "test"]
SENSOR_TYPES = ["primesense"]# , "kinect", "canon"]
MODEL_TYPES = ["cad", "cad_subdivided", "reconst"]

def get_console_width() -> int:
    """Return width of available window area."""
    try:
        import fcntl
        import termios
        import struct
        return struct.unpack('hh', fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, '1234'))[1]
    except:
        return 80

def bar_thermometer(current: int, total: int, width: int = 80) -> str:
    """Return thermometer style progress bar string."""
    if total <= 0:
        return '[error]'
    avail_dots = width - 2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    return '[' + '.'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'

def bar_adaptive(current: int, total: int, width: int = 80) -> str:
    """Return progress bar string with adaptive layout."""
    if not total or total < 0:
        msg = f"{current} / unknown"
        return msg if len(msg) < width else str(current)

    min_width = {
        'percent': 4,
        'bar': 3,
        'size': len(f"{total}") * 2 + 3,
    }
    priority = ['percent', 'bar', 'size']

    selected = []
    avail = width
    for field in priority:
        if min_width[field] < avail:
            selected.append(field)
            avail -= min_width[field] + 1

    output = []
    for field in selected:
        if field == 'percent':
            output.append(f"{100 * current // total}%".rjust(min_width['percent']))
        elif field == 'bar':
            output.append(bar_thermometer(current, total, min_width['bar'] + avail))
        elif field == 'size':
            output.append(f"{current} / {total}".rjust(min_width['size']))

    return ' '.join(output)

def callback_progress(blocks: int, block_size: int, total_size: int, bar_function=bar_adaptive):
    """Callback function for urlretrieve to show download progress."""
    current_size = min(blocks * block_size, total_size)
    progress = bar_function(current_size, total_size, min(100, get_console_width()))
    if progress:
        sys.stdout.write("\r" + progress)
        sys.stdout.flush()

def download_file(url: str, output_path: str, show_progress: bool = True) -> str:
    """Download a file from URL to the specified path."""
    import urllib.request
    
    if show_progress:
        urllib.request.urlretrieve(url, output_path, 
                                 lambda b, bs, ts: callback_progress(b, bs, ts))
    else:
        urllib.request.urlretrieve(url, output_path)
    
    return output_path

def unzip_file(zip_path: str, extract_path: str, show_progress: bool = True):
    """Unzip file with progress indication."""
    with zipfile.ZipFile(zip_path) as zf:
        if show_progress:
            total_size = sum(f.file_size for f in zf.filelist)
            extracted_size = 0
            for file in zf.filelist:
                zf.extract(file, extract_path)
                extracted_size += file.file_size
                callback_progress(extracted_size, total_size, total_size)
        else:
            zf.extractall(extract_path)
    
    os.remove(zip_path)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Downloads and unpacks the selected parts of the T-LESS dataset."
    )
    parser.add_argument("--destination", default=".",
                       help="Destination folder (default: current directory)")
    parser.add_argument("--train", nargs="*", type=int, choices=TRAIN_IDS,
                       metavar="obj_id", help="Training object IDs")
    parser.add_argument("--test", nargs="*", type=int, choices=TEST_IDS,
                       metavar="scene_id", help="Test scene IDs")
    parser.add_argument("--sensors", nargs="+", choices=SENSOR_TYPES,
                       default=SENSOR_TYPES, help="Sensor types")
    parser.add_argument("--models", nargs="*", choices=MODEL_TYPES,
                       default=MODEL_TYPES, help="3D model variants")
    
    args = parser.parse_args()
    
    # Handle default values
    args.train = args.train if args.train is not None else TRAIN_IDS
    args.test = args.test if args.test is not None else TEST_IDS
    args.models = args.models if args.models is not None else MODEL_TYPES
    
    return args

def main():
    args = parse_arguments()
    dest_path = os.path.abspath(args.destination)
    dataset_path = os.path.join(dest_path, f"t-less_v{VERSION}")
    
    # Create destination directory
    os.makedirs(dataset_path, exist_ok=True)
    
    # Calculate total steps for progress tracking
    total_steps = (len(args.sensors) * (len(args.train) + len(args.test)) + 
                  len(args.models))
    current_step = 0
    
    print(f"\nDownloading T-LESS dataset v{VERSION} to: {dataset_path}")
    print(f"Total downloads: {total_steps}\n")
    
    # Download training and test data
    for sensor in args.sensors:
        # Training data
        # for obj_id in args.train:
        #     current_step += 1
        #     print(f"\nDownloading training data ({current_step}/{total_steps})")
        #     url = f"{URL_ROOT}/t-less_v{VERSION}_train_{sensor}_{obj_id:02d}.zip"
        #     output_path = os.path.join(dataset_path, f"train_{sensor}_{obj_id:02d}.zip")
        #     download_file(url, output_path)
        #     unzip_file(output_path, dataset_path)
        
        # Test data
        for scene_id in args.test:
            current_step += 1
            print(f"\nDownloading test data ({current_step}/{total_steps})")
            url = f"{URL_ROOT}/t-less_v{VERSION}_test_{sensor}_{scene_id:02d}.zip"
            output_path = os.path.join(dataset_path, f"test_{sensor}_{scene_id:02d}.zip")
            download_file(url, output_path)
            unzip_file(output_path, dataset_path)
    
    # Download models
    for model_type in args.models:
        current_step += 1
        print(f"\nDownloading models ({current_step}/{total_steps})")
        url = f"{URL_ROOT}/t-less_v{VERSION}_models_{model_type}.zip"
        output_path = os.path.join(dataset_path, f"models_{model_type}.zip")
        download_file(url, output_path)
        unzip_file(output_path, dataset_path)
    
    print("\nDownload complete!")

if __name__ == '__main__':
    main()