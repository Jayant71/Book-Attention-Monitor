# Attention Monitoring System

A real-time attention monitoring system that uses AWS Rekognition to analyze user attention levels through webcam feed.

## Features

- Real-time attention monitoring using webcam
- Face detection and pose analysis using AWS Rekognition
- Attention metrics calculation and session reporting
- Configurable monitoring duration and check intervals
- Detailed session logs with timestamps

## Prerequisites

- Python 3.8+
- AWS Account with Rekognition access
- Webcam
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/attention-monitoring-system.git
cd attention-monitoring-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your AWS credentials:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

## Project Structure

```
attention-monitoring-system/
├── src/
│   ├── analysis/
│   │   └── attention_analyzer.py
│   ├── aws/
│   │   └── rekognition_client.py
│   ├── camera/
│   │   └── camera_manager.py
│   ├── session/
│   │   └── session_manager.py
│   └── main.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Usage

Run the main script to start monitoring:

```bash
python src/main.py
```

The system will:
1. Initialize the webcam
2. Start monitoring attention levels
3. Display real-time status
4. Generate a session report upon completion

## Configuration

- Default session duration: 30 minutes
- Check interval: 2 seconds
- Attention thresholds:
  - Yaw (left-right head rotation): ±30 degrees
  - Pitch (up-down head rotation): ±20 degrees

These values can be modified in `attention_analyzer.py`.

## Session Reports

Reports are saved as JSON files containing:
- Session start and end times
- Attention metrics
- Detailed attention logs with timestamps

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

[MIT License](LICENSE)

## Authors

- Your Name - Initial work

## Acknowledgments

- AWS Rekognition for face detection
- OpenCV for camera handling