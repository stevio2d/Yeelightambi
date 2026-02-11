# Yeelight Ambilight System

A real-time ambient lighting system that captures your screen's dominant colors and syncs them to a Yeelight smart bulb for an immersive ambilight experience.

## Features

- **Real-time Color Sync**: Captures screen content and extracts dominant colors using advanced K-means clustering
- **High Performance**: Multi-threaded architecture with optimized screen capture (5x faster than PIL)
- **Smooth Transitions**: Color smoothing and saturation enhancement for vibrant, fluid lighting
- **Smart Filtering**: Excludes UI elements and dark pixels for better color accuracy
- **Configurable**: Command-line options for brightness, frequency, color processing, and more
- **Robust**: Graceful error handling, logging, and clean shutdown
- **Secure**: All processing in memory - no sensitive screen data saved to disk

## Requirements

### Hardware
- Yeelight smart bulb (with LAN control enabled)
- Computer with display output

### Software Dependencies
```bash
pip install yeelight numpy scikit-learn opencv-python mss webcolors
```

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install yeelight numpy scikit-learn opencv-python mss webcolors
   ```

## Setup

### Yeelight Bulb Configuration
1. Install the Yeelight app and connect your bulb
2. Enable "LAN Control" in the bulb settings
3. Note your bulb's IP address (found in the app)

### Network Setup
- Ensure your computer and Yeelight bulb are on the same network
- The system uses port 12345 for music mode communication

## Usage

### Basic Usage
```bash
python3 yeelightambi.py --bulb-ip 192.168.1.100
```

### Advanced Configuration
```bash
python3 yeelightambi.py \
  --bulb-ip 192.168.1.100 \
  --brightness 80 \
  --frequency 0.1 \
  --clusters 5 \
  --saturation 1.5
```

### Command Line Options
- `--bulb-ip`: IP address of your Yeelight bulb (required)
- `--brightness`: Bulb brightness 1-100 (default: 100)
- `--frequency`: Update frequency in seconds (default: 0.2)
- `--clusters`: Number of color clusters for analysis (default: 3)
- `--saturation`: Color saturation boost multiplier (default: 1.2)

### Using the Startup Script
Edit `yeelightstart.sh` to set your correct paths and run:
```bash
chmod +x yeelightstart.sh
./yeelightstart.sh
```

## How It Works

1. **Screen Capture**: Uses `mss` library for fast, efficient screen capture
2. **Image Processing**: Crops UI elements and resizes for optimal performance
3. **Color Analysis**: K-means clustering identifies the most dominant colors
4. **Color Enhancement**: HSV-based saturation boosting and smooth transitions
5. **Bulb Communication**: Updates Yeelight via music mode for minimal latency

## Performance Optimization

- **Threading**: Separate capture and processing threads prevent blocking
- **Frame Skipping**: Drops frames when processing can't keep up
- **Smart Cropping**: Excludes window borders and taskbars (configurable)
- **Efficient Clustering**: Adaptive cluster count based on image complexity
- **Memory Only**: No disk I/O for maximum speed and security

## Troubleshooting

### Connection Issues
- Verify bulb IP address with Yeelight app
- Ensure LAN Control is enabled on the bulb
- Check that computer and bulb are on same network
- Try disabling firewall temporarily

### Performance Issues
- Increase update frequency (lower --frequency value)
- Reduce color clusters (--clusters 1-2)
- Check CPU usage - the system is optimized but intensive

### Color Accuracy
- Adjust saturation boost (--saturation)
- Modify UI cropping in configuration
- Experiment with different cluster counts

## Configuration

The system uses a configurable architecture with three levels of configuration:

### 1. config.py File
Default settings are defined in `config.py`. Edit this file to change system defaults:

```python
DEFAULT_BULB_IP = "192.168.50.162"      # Default bulb IP
DEFAULT_MUSIC_MODE_PORT = 12345         # Music mode communication port
DEFAULT_BRIGHTNESS = 100                # Bulb brightness (1-100)
DEFAULT_UPDATE_FREQUENCY = 0.2          # Update interval in seconds
# ... and many more settings
```

### 2. Environment Variables
Override defaults using environment variables:

```bash
export YEELIGHT_BULB_IP="192.168.1.100"
export YEELIGHT_BRIGHTNESS=80
export AMBILIGHT_UPDATE_FREQ=0.1
```

### 3. Command Line Arguments
Override both defaults and environment variables:

```bash
python3 yeelightambi.py --bulb-ip 192.168.1.100 --brightness 80 --frequency 0.1
```

## Safety & Privacy

- **No Data Storage**: Screenshots are processed in memory only
- **Local Processing**: All analysis happens on your computer
- **Network Security**: Only RGB values sent to bulb, no screen content
- **Graceful Shutdown**: Ctrl+C cleanly stops all processes

## Contributing

Feel free to submit issues and enhancement requests. When contributing:

1. Follow the existing code style and type hints
2. Test thoroughly with different screen content
3. Ensure no performance regressions
4. Update documentation as needed

## License

This project is open source. Use responsibly and ensure your Yeelight usage complies with local regulations.