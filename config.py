"""
Configuration settings for the Yeelight Ambilight System.

This file contains default values that can be overridden by command-line arguments
or environment variables. Modify these values to customize your setup.
"""

import os

# Yeelight Bulb Configuration
DEFAULT_BULB_IP = "192.168.50.162"
DEFAULT_MUSIC_MODE_PORT = 12345

# Display and Performance Settings
DEFAULT_BRIGHTNESS = 100
DEFAULT_UPDATE_FREQUENCY = 0.2  # seconds between updates
DEFAULT_SCREEN_WIDTH = 200      # processing resolution width
DEFAULT_SCREEN_HEIGHT = 150     # processing resolution height

# Color Processing Settings
DEFAULT_N_CLUSTERS = 3              # K-means color clusters
DEFAULT_MIN_BRIGHTNESS_THRESHOLD = 20   # filter dark pixels below this value
DEFAULT_COLOR_SMOOTHING = 0.3       # 0-1, higher = more smoothing between frames
DEFAULT_SATURATION_BOOST = 1.2      # color saturation multiplier

# UI Filtering Settings
DEFAULT_CROP_UI_ELEMENTS = True     # remove screen edges (taskbars, borders)
DEFAULT_UI_CROP_PERCENT = 10        # percentage to crop from each edge

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Environment Variable Overrides
# These can be set in your shell environment to override defaults
BULB_IP = os.getenv("YEELIGHT_BULB_IP", DEFAULT_BULB_IP)
MUSIC_MODE_PORT = int(os.getenv("YEELIGHT_MUSIC_PORT", DEFAULT_MUSIC_MODE_PORT))
BRIGHTNESS = int(os.getenv("YEELIGHT_BRIGHTNESS", DEFAULT_BRIGHTNESS))
UPDATE_FREQUENCY = float(os.getenv("AMBILIGHT_UPDATE_FREQ", DEFAULT_UPDATE_FREQUENCY))

# Advanced Settings (rarely need to be changed)
QUEUE_MAX_SIZE = 2                  # screen capture queue size
CAPTURE_THREAD_SLEEP_MULTIPLIER = 0.5   # capture frequency relative to processing
NETWORK_TIMEOUT_RETRY_DELAY = 0.1   # seconds to wait after network errors
DEFAULT_DIM_COLOR = [50, 50, 50]    # fallback color when screen is too dark

# Performance Optimization Settings
KMEANS_RANDOM_STATE = 42            # for consistent color clustering results
KMEANS_N_INIT = 1                   # single initialization for speed
HSV_SATURATION_MAX = 255            # maximum saturation value in HSV space
RGB_VALUE_MAX = 255                 # maximum RGB component value
RGB_VALUE_MIN = 0                   # minimum RGB component value