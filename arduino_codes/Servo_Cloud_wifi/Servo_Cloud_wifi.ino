#include <WiFi.h>
#include <WiFiUdp.h>
#include <NTPClient.h>
#include <ESP32Servo.h>
#include <FirebaseESP32.h>



// network credentials(naeme & pass)
#define WIFI_SSID "Ssmn"
#define WIFI_PASSWORD "ghpw2180t78ujl"

// Firebase settings
#define FIREBASE_HOST "smb19909-304e0-default-rtdb.firebaseio.com/"  // Firebase project URL
#define FIREBASE_AUTH "dMskWLsXdvTEMroHgfXLNldrPVvVNFQXVZinbItn"  //  Firebase Database Secret

Servo myservo;
const int servoPin = D3;  // Pin attached to D3

// NTP client setup
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 3600);  // Set timezone to UTC (UK time is UTC+0)

// Firebase setup
FirebaseData feedData, timerData;
FirebaseAuth auth;              // Firebase Authentication
FirebaseConfig config;           // Firebase Config

int feedNow = 0;
String scheduledTimers[3] = {"00:00", "00:00", "00:00"};
String currentTimeString;

void setup() {
  Serial.begin(115200);
  
  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nConnected to Wi-Fi");

  // Initialize Firebase
  config.host = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;  // Using legacy token for Firebase authentication
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  // Check if Firebase is ready
  if (Firebase.ready()) {
    Serial.println("Firebase initialized successfully");
  } else {
    Serial.println("Failed to initialize Firebase");
  }

  // Attach the servo
  myservo.attach(servoPin);

  // Initialize NTP client to get current time
  timeClient.begin();
}

void loop() {
  // Update the NTP time
  timeClient.update();
  
  // Get the current time in hours and minutes
  currentTimeString = String(timeClient.getHours()) + ":" + String(timeClient.getMinutes());
  Serial.print("Current Time: ");
  Serial.println(currentTimeString);

  // Check for direct feed request
  if (Firebase.getInt(feedData, "/feednow")) {
    // Now retrieve the value directly from feedData
    feedNow = feedData.intData();
    Serial.print("FeedNow Status: ");
    Serial.println(feedNow);

    if (feedNow == 1) {
      feedFish();
      feedNow = 0;
      Firebase.setInt(feedData, "/feednow", feedNow);  // Reset feedNow after feeding
      Serial.println("Fish Fed Successfully");
    }
  } else {
    // Log any error in retrieving the feedNow value
    Serial.println("Failed to retrieve feedNow from Firebase.");
    Serial.println(feedData.errorReason());
  }

  // Retrieve and check scheduled feed times
  for (int i = 0; i < 3; i++) {
    String path = "/timers/timer" + String(i);
    Serial.print("Trying to retrieve data from: ");
    Serial.println(path);  // Print the exact path being accessed

    if (Firebase.getString(timerData, path)) {
      scheduledTimers[i] = timerData.stringData();  // Full string data
      Serial.print("Scheduled Timer " + String(i) + ": ");
      Serial.println(scheduledTimers[i]);
    } else {
      // Log error for each specific path and set default value to avoid crashing the program
      Serial.print("Failed to retrieve scheduled timer from path: ");
      Serial.println(path);
      Serial.println(timerData.errorReason());  // Print Firebase error message
      scheduledTimers[i] = "00:00";  // Set a default time to avoid issues
    }
  }

  // Check if it's time to feed based on the schedule
  if (currentTimeString == scheduledTimers[0] || currentTimeString == scheduledTimers[1] || currentTimeString == scheduledTimers[2]) {
    feedNow = 1;  // Set feedNow to 1 when the scheduled feeding time is reached
    Firebase.setInt(feedData, "/feednow", feedNow);  // Update feedNow status in Firebase
    feedFish();
    delay(60000);  // Wait 1 minute to prevent multiple triggers within the same minute
  }

  delay(1000);  // Check every second
}

void feedFish() {
  Serial.println("Feeding the fish...");
  myservo.writeMicroseconds(3000);  // Rotate clockwise
  delay(3000);                       // Rotate for 700 ms (adjust as needed)
  myservo.writeMicroseconds(1500);  // Stop rotation
  delay(1000);                      // Wait for servo to reset
  Serial.println("Feeding complete.");
}
