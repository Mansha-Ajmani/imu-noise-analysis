# IMU Noise Analysis for Robotics Applications

## Overview
This project analyzes IMU (Inertial Measurement Unit) sensor noise using Python.  
The analysis compares gyroscope and accelerometer behavior across different environments to study sensor reliability in robotics applications.

## Features
- Data preprocessing and timestamp alignment
- Gyroscope noise analysis
- Accelerometer noise analysis
- Statistical summaries
- Time-series visualization
- Density heatmaps using KDE
- Sensor correlation analysis

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- SciPy

## Project Structure
```bash
imu-noise-analysis/
│
├── data/
├── outputs/
├── imu_noise_analysis.py
├── requirements.txt
└── README.md
```

## Visualizations

### Gyroscope Noise Boxplot
![Gyroscope Boxplot](outputs/plot1_gyro_boxplot.png)

### Accelerometer Noise Comparison
![Accelerometer Boxplot](outputs/plot2_acc_boxplot.png)

### Noise Variation Over Time
![Noise Variation](outputs/plot3_noise_over_time.png)

### Gyroscope Density Heatmap
![Density Heatmap](outputs/plot4_gyro_density.png)

### Sensor Correlation Heatmap
![Correlation Heatmap](outputs/plot5_sensor_correlation.png)

## How to Run

Clone the repository:
```bash
git clone https://github.com/yourusername/imu-noise-analysis.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the script:
```bash
python imu_noise_analysis.py
```

## Applications
- Robotics
- Motion tracking
- Autonomous systems
- Sensor reliability analysis
- IMU data processing

## Author
Mansha Ajmani
