#include <OneWire.h>
#include <DallasTemperature.h>
#include <NTPClient.h>
#include <FirebaseESP32.h>

// Replace with your network credentials(naeme & pass)
const char* ssid = "";
const char* password =  "";

// Firebase settings
// Firebase configuration
FirebaseConfig config;
FirebaseAuth auth;
String firebaseHost = "";  // Replace with your Firebase project URL
String firebaseAuth =  "";  // Replace with your Firebase Database Secret



// Pin connected to the Data pin of the DS18B20
#define ONE_WIRE_BUS D2

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// Firebase object
FirebaseData firebaseData;

// Timekeeping
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 3600, 60000); // Update time every minute

// Specify the days to send data (0 = Sunday, 1 = Monday, etc.)
int validDays[3] = {1, 3, 5};  // Example: Monday, Wednesday, Friday

// Function to check if the current day is one of the valid days (e.g., Monday, Wednesday, Friday)
bool isValidDay(int currentDay) {
  for (int i = 0; i < 3; i++) {
    if (validDays[i] == currentDay) {
      return true;
    }
  }
  return false;
}

void setup() {
  // Start Serial communication
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to Wi-Fi");

  // Set Firebase host and authentication
  config.host = firebaseHost.c_str();
  config.signer.tokens.legacy_token = firebaseAuth.c_str();

  // Initialize Firebase
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  // Start DallasTemperature library
  sensors.begin();

  // Initialize NTP client
  timeClient.begin();
}
void loop() {
  timeClient.update();
  unsigned long epochTime = timeClient.getEpochTime();

  Serial.print("Epoch Time: ");
  Serial.println(epochTime);

  int currentHour = timeClient.getHours();
  int currentDay = timeClient.getDay();  // 0 = Sunday, 1 = Monday, etc.
  int currentMinute = timeClient.getMinutes();

  Serial.print("Current Day: ");
  Serial.println(currentDay);
  Serial.print("Current Hour: ");
  Serial.println(currentHour);
  Serial.print("Current Minute: ");
  Serial.println(currentMinute);

  // Check if today is a valid day (e.g., Monday, Wednesday, Friday)
  if (isValidDay(currentDay)) {
    Serial.println("Today is a valid day for sending data.");

    // Send data at specific times of the day (8 AM, 12 PM, 8 PM)
    if (currentHour == 8 || currentHour == 20 || currentHour == 19 && currentMinute == 12) {
      // Request temperature from DS18B20 sensor
      sensors.requestTemperatures();
      float temperatureC = sensors.getTempCByIndex(0);

      if (temperatureC != DEVICE_DISCONNECTED_C) {
        Serial.print("Temperature: ");
        Serial.print(temperatureC);
        Serial.println(" °C");

        // Push temperature data to Firebase
        if (Firebase.pushFloat(firebaseData, "/temperature", temperatureC)) {
          Serial.println("Temperature sent to Firebase successfully");
        } else {
          Serial.print("Error sending data: ");
          Serial.println(firebaseData.errorReason());
        }
      } else {
        Serial.println("Error: Could not read temperature data. Is the sensor connected?");
      }

      // Wait an hour to avoid resending within the same hour
      delay(3600000);  // 1 hour delay
    }
  } else {
    Serial.println("Today is not a valid day.");
  }

  // Delay before next loop iteration (1 minute)
  delay(60000);
}
