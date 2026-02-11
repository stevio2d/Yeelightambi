from yeelight import Bulb
import numpy as np
import time
from sklearn.cluster import KMeans
import webcolors
import socket
import cv2
import mss
import logging
import signal
import sys
import os
import argparse
from typing import Tuple, Optional, List
from dataclasses import dataclass
import threading
from queue import Queue, Empty

import config

@dataclass
class AmbiLightConfig:
    """Configuration settings for the ambilight system."""
    bulb_ip: str = config.BULB_IP
    brightness: int = config.BRIGHTNESS
    update_frequency: float = config.UPDATE_FREQUENCY
    screen_width: int = config.DEFAULT_SCREEN_WIDTH
    screen_height: int = config.DEFAULT_SCREEN_HEIGHT
    n_clusters: int = config.DEFAULT_N_CLUSTERS
    min_brightness_threshold: int = config.DEFAULT_MIN_BRIGHTNESS_THRESHOLD
    color_smoothing: float = config.DEFAULT_COLOR_SMOOTHING
    saturation_boost: float = config.DEFAULT_SATURATION_BOOST
    crop_ui_elements: bool = config.DEFAULT_CROP_UI_ELEMENTS
    ui_crop_percent: int = config.DEFAULT_UI_CROP_PERCENT
    
    @classmethod
    def from_args(cls) -> 'AmbiLightConfig':
        """Create config from command line arguments."""
        parser = argparse.ArgumentParser(description='Yeelight Ambilight System')
        parser.add_argument('--bulb-ip', default=cls.bulb_ip, help='Yeelight bulb IP address')
        parser.add_argument('--brightness', type=int, default=cls.brightness, help='Bulb brightness (1-100)')
        parser.add_argument('--frequency', type=float, default=cls.update_frequency, help='Update frequency in seconds')
        parser.add_argument('--clusters', type=int, default=cls.n_clusters, help='Number of color clusters')
        parser.add_argument('--saturation', type=float, default=cls.saturation_boost, help='Color saturation boost')
        
        args = parser.parse_args()
        return cls(
            bulb_ip=args.bulb_ip,
            brightness=args.brightness,
            update_frequency=args.frequency,
            n_clusters=args.clusters,
            saturation_boost=args.saturation
        )

def setup_logging() -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, config.DEFAULT_LOG_LEVEL.upper())
    logging.basicConfig(
        level=log_level,
        format=config.DEFAULT_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_color_name(rgb_tuple: Tuple[int, int, int]) -> str:
    """Get the closest color name for an RGB tuple."""
    hex_color = webcolors.rgb_to_hex(rgb_tuple)
    
    try:
        return webcolors.hex_to_name(hex_color)
    except ValueError:
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb_tuple[0]) ** 2
            gd = (g_c - rgb_tuple[1]) ** 2
            bd = (b_c - rgb_tuple[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

def enhance_color_saturation(rgb: List[int], boost: float) -> List[int]:
    """Enhance color saturation using HSV color space."""
    # Convert RGB to HSV
    rgb_normalized = np.array(rgb) / config.RGB_VALUE_MAX
    hsv = cv2.cvtColor(np.uint8([[rgb_normalized * config.RGB_VALUE_MAX]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Boost saturation
    hsv[1] = min(config.HSV_SATURATION_MAX, hsv[1] * boost)
    
    # Convert back to RGB
    rgb_enhanced = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]
    return [int(c) for c in rgb_enhanced]

def smooth_color_transition(current: List[int], previous: List[int], smoothing: float) -> List[int]:
    """Apply exponential moving average for smooth color transitions."""
    if previous is None:
        return current
    
    smoothed = []
    for i in range(3):
        smoothed.append(int(current[i] * (1 - smoothing) + previous[i] * smoothing))
    return smoothed

def crop_ui_elements(image: np.ndarray, crop_percent: int) -> np.ndarray:
    """Crop UI elements from screen edges."""
    if crop_percent <= 0:
        return image
    
    h, w = image.shape[:2]
    crop_h = int(h * crop_percent / 100)
    crop_w = int(w * crop_percent / 100)
    
    return image[crop_h:h-crop_h, crop_w:w-crop_w]

def extract_dominant_colors(image: np.ndarray, config: AmbiLightConfig) -> List[int]:
    """Extract dominant color from image using improved K-means clustering."""
    # Crop UI elements if enabled
    if config.crop_ui_elements:
        image = crop_ui_elements(image, config.ui_crop_percent)
    
    # Resize for performance
    resized = cv2.resize(image, (config.screen_width, config.screen_height))
    reshaped = resized.reshape(-1, 3)
    
    # Remove very dark pixels
    mask = np.sum(reshaped, axis=1) > config.min_brightness_threshold
    filtered_pixels = reshaped[mask]
    
    if len(filtered_pixels) == 0:
        return config.DEFAULT_DIM_COLOR
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=min(config.n_clusters, len(filtered_pixels)), 
        n_init=config.KMEANS_N_INIT, 
        random_state=config.KMEANS_RANDOM_STATE
    )
    kmeans.fit(filtered_pixels)
    
    # Get cluster centers and their sizes
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Find the largest cluster (most common color)
    unique, counts = np.unique(labels, return_counts=True)
    dominant_cluster_idx = unique[np.argmax(counts)]
    dominant_color = centers[dominant_cluster_idx]
    
    # Clamp values and convert to int
    dominant_color = [max(config.RGB_VALUE_MIN, min(config.RGB_VALUE_MAX, int(c))) for c in dominant_color]
    
    # Enhance saturation
    if config.saturation_boost > 1.0:
        dominant_color = enhance_color_saturation(dominant_color, config.saturation_boost)
    
    return dominant_color

def capture_screen_worker(screen_queue: Queue, config: AmbiLightConfig, stop_event: threading.Event) -> None:
    """Worker thread for screen capture."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        
        while not stop_event.is_set():
            try:
                # Capture screen
                screenshot = sct.grab(monitor)
                img_array = np.array(screenshot)[:, :, :3]  # Remove alpha channel
                
                # Put in queue (non-blocking)
                try:
                    screen_queue.put_nowait(img_array)
                except:
                    pass  # Queue full, skip this frame
                    
                time.sleep(config.update_frequency * config.CAPTURE_THREAD_SLEEP_MULTIPLIER)
                
            except Exception as e:
                logging.error(f"Screen capture error: {e}")
                time.sleep(config.NETWORK_TIMEOUT_RETRY_DELAY)

class AmbiLightSystem:
    """Main ambilight system class."""
    
    def __init__(self, config: AmbiLightConfig):
        self.config = config
        self.bulb: Optional[Bulb] = None
        self.previous_color: Optional[List[int]] = None
        self.running = False
        self.screen_queue = Queue(maxsize=config.QUEUE_MAX_SIZE)
        self.stop_event = threading.Event()
        
    def setup_bulb(self) -> bool:
        """Initialize and configure the Yeelight bulb."""
        try:
            self.bulb = Bulb(self.config.bulb_ip)
            self.bulb.set_brightness(self.config.brightness)
            
            # Get local IP for music mode
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Start music mode for faster updates
            properties = self.bulb.get_properties()
            if properties.get('music_on') == '0':
                self.bulb.start_music(port=config.MUSIC_MODE_PORT, ip=local_ip)
                logging.info("Music mode enabled for faster color updates")
            
            logging.info(f"Connected to Yeelight bulb at {self.config.bulb_ip}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup bulb: {e}")
            return False
    
    def signal_handler(self, signum, frame):
        """Handle graceful shutdown."""
        logging.info("Received shutdown signal, stopping ambilight system...")
        self.stop()
        
    def start(self) -> None:
        """Start the ambilight system."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        if not self.setup_bulb():
            return
            
        self.running = True
        
        # Start screen capture thread
        capture_thread = threading.Thread(
            target=capture_screen_worker,
            args=(self.screen_queue, self.config, self.stop_event)
        )
        capture_thread.daemon = True
        capture_thread.start()
        
        logging.info("Ambilight system started. Press Ctrl+C to stop.")
        
        # Main processing loop
        while self.running:
            try:
                # Get latest screen capture
                try:
                    screen_image = self.screen_queue.get(timeout=self.config.update_frequency)
                except Empty:
                    continue
                
                # Extract dominant color
                dominant_color = extract_dominant_colors(screen_image, self.config)
                
                # Apply color smoothing
                if self.config.color_smoothing > 0:
                    dominant_color = smooth_color_transition(
                        dominant_color, self.previous_color, self.config.color_smoothing
                    )
                
                # Update bulb color
                if self.bulb:
                    self.bulb.set_rgb(dominant_color[0], dominant_color[1], dominant_color[2])
                
                self.previous_color = dominant_color
                
                # Log color info (optional)
                color_name = get_color_name(tuple(dominant_color))
                logging.debug(f"Color: {color_name} RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})")
                
            except Exception as e:
                logging.error(f"Processing error: {e}")
                time.sleep(config.NETWORK_TIMEOUT_RETRY_DELAY)
    
    def stop(self) -> None:
        """Stop the ambilight system."""
        self.running = False
        self.stop_event.set()
        logging.info("Ambilight system stopped")

def main():
    """Main entry point."""
    setup_logging()
    config = AmbiLightConfig.from_args()
    
    system = AmbiLightSystem(config)
    system.start()

if __name__ == "__main__":
    main()
