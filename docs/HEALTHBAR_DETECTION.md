# Roboflow Health Bar Detection Integration

This bot now supports automatic health bar detection using a pre-trained Roboflow computer vision model.

## Setup Instructions

### 1. Install Required Package
```bash
pip install inference-sdk
```

### 2. Get Your API Key
- Visit https://app.roboflow.com and create a free account
- Your API key appears in Account → API Keys
- Copy your API key

### 3. Configure the Bot
Edit `config.py` and update:
```python
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"
ROBOFLOW_HEALTHBAR_ENABLED = True
```

## Usage

The health bar detector is automatically initialized when the bot starts if enabled.

### In Your Code

Use the `detect_healthbars()` function to detect health bars in a frame:

```python
from detectors import detect_healthbars

# In your frame processing loop
detections = detect_healthbars(frame_bgr)
for detection in detections:
    x = detection['x']  # center x
    y = detection['y']  # center y
    width = detection['width']
    height = detection['height']
    confidence = detection['confidence']
    print(f"Health bar at ({x}, {y}) with confidence {confidence}")
```

## Model Details

- **Model**: Hero Siege Healthbars by discordturtle
- **Accuracy**: 97.8% mAP@50, 96.2% Precision, 97.5% Recall
- **Classes**: Detects "healthbars" class only
- **License**: CC BY 4.0

## Cost

The Roboflow API is **free** for inference on public models like this one. No costs are incurred for using this integration.

## Performance

- Network request time: ~100-500ms (can be cached)
- Detector automatically caches workflow definition for 15 minutes
- Results are reliable for detecting enemy/NPC health bars on screen

## Troubleshooting

If the detector doesn't work:
1. Verify `ROBOFLOW_HEALTHBAR_ENABLED = True` in config.py
2. Verify your API key is correct in config.py
3. Check the console for `[HEALTHBAR]` log messages
4. Ensure `inference-sdk` is installed: `pip list | grep inference`
5. Test your API key at https://universe.roboflow.com/discordturtle/hero-siege-healthbars-8gez6
