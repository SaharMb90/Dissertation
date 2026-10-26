# Smart Aquarium: Closed-Loop IoT, Computer Vision and ML System

Code for my MSc dissertation in Advanced Computer Science at Liverpool Hope University (2025). The project combines ESP32 firmware, a cloud database, image processing and machine learning to monitor and automate a home aquarium.

## What is in this repository

```
arduino_codes/
  Servo_Cloud_wifi/Servo_Cloud_wifi.ino   ESP32 fish feeder: servo motor driven by Firebase
  temp_wifi/temp_wifi.ino                 ESP32 temperature logger: DS18B20 sensor to Firebase
ML & ImageProcessing_code/
  data.csv                                Water-quality log, 61 daily records (Aug–Sep 2024)
  predictModel.py                         Compares six regression models
  extra.py                                Searches for the best predicted water conditions
  SecondMethod.py                         OpenCV fish segmentation and size estimation
```

## 1. IoT firmware (ESP32, C++)

**Automated feeder (`Servo_Cloud_wifi.ino`).** The ESP32 connects to Wi-Fi, reads the current time from an NTP server, and polls a Firebase Realtime Database. It turns a servo motor to feed the fish either immediately (when the `/feednow` flag is set from the web app) or at up to three scheduled times stored under `/timers`, then resets the flag.

**Temperature logger (`temp_wifi.ino`).** A DS18B20 waterproof sensor reads the water temperature on selected days and times and pushes each reading to Firebase, so it can be displayed in the web app and used for analysis.

Libraries: `WiFi`, `NTPClient`, `ESP32Servo`, `FirebaseESP32`, `OneWire`, `DallasTemperature`.

To use the firmware, open the `.ino` file in the Arduino IDE with ESP32 board support installed, then enter your own Wi-Fi name and password and your Firebase URL and database secret before uploading.

## 2. Machine learning (Python)

`data.csv` holds daily water-quality measurements: temperature, total dissolved solids (TDS), electrical conductivity (EC), light intensity (low / medium / high), survival rate and pH.

`predictModel.py` scales the features, holds out 20% of the data for testing, and compares six regressors: Random Forest (tuned with 5-fold grid search), Gradient Boosting, Bagging with decision trees, Stacking (Random Forest + Decision Tree), Linear Regression and Support Vector Regression. It reports MSE and R² for each.

`extra.py` trains the Bagging model and samples 100 combinations of temperature, TDS, pH and light intensity to find the conditions with the highest predicted growth.

**Important limitation.** Fish growth could not be measured reliably during the two-month data collection, so the growth-rate target in both scripts is **simulated** from random values. The scripts therefore demonstrate the modelling pipeline (preprocessing, tuning, model comparison and condition search) rather than real predictive results on fish growth.

## 3. Image processing (OpenCV)

`SecondMethod.py` estimates fish sizes from a photo of the tank:

1. Convert the image to HSV and apply a colour mask for the fish.
2. Clean the mask with Gaussian blur, erosion and dilation (morphological operations).
3. Find contours and use the largest fish, of known length, as the scale reference (pixels per mm).
4. Estimate the area of every other fish in mm² and draw labelled contours on the image.

Before running it, change `image_path` at the bottom of the file to an image on your computer, and set `known_fish_size_mm` to the real length of the reference fish. Adjust the HSV bounds if your fish are a different colour.

## Running the Python code

With Anaconda (as used in the project): open Anaconda Navigator, launch Spyder or Jupyter, and run the scripts from inside `ML & ImageProcessing_code/` so that `data.csv` is found.

Or with pip:

```bash
pip install pandas numpy scikit-learn opencv-python matplotlib
cd "ML & ImageProcessing_code"
python predictModel.py
python extra.py
python SecondMethod.py
```

## Author

Sahar Mirzabaki · [GitHub](https://github.com/SaharMb90) · [Portfolio](https://portfolio2-phi-khaki.vercel.app/)
