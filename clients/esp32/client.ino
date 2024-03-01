// #include <esp_wifi.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <ArduinoWebsockets.h>
#include <queue>

// Define a global queue to hold the audio data messages.
std::queue<std::vector<uint8_t>> audioQueue;

using namespace websockets;

const char* rootCACertificate PROGMEM = \
"-----BEGIN CERTIFICATE-----\n" \
"MIIFFjCCAv6gAwIBAgIRAJErCErPDBinU/bWLiWnX1owDQYJKoZIhvcNAQELBQAw\n" \
"TzELMAkGA1UEBhMCVVMxKTAnBgNVBAoTIEludGVybmV0IFNlY3VyaXR5IFJlc2Vh\n" \
"cmNoIEdyb3VwMRUwEwYDVQQDEwxJU1JHIFJvb3QgWDEwHhcNMjAwOTA0MDAwMDAw\n" \
"WhcNMjUwOTE1MTYwMDAwWjAyMQswCQYDVQQGEwJVUzEWMBQGA1UEChMNTGV0J3Mg\n" \
"RW5jcnlwdDELMAkGA1UEAxMCUjMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK\n" \
"AoIBAQC7AhUozPaglNMPEuyNVZLD+ILxmaZ6QoinXSaqtSu5xUyxr45r+XXIo9cP\n" \
"R5QUVTVXjJ6oojkZ9YI8QqlObvU7wy7bjcCwXPNZOOftz2nwWgsbvsCUJCWH+jdx\n" \
"sxPnHKzhm+/b5DtFUkWWqcFTzjTIUu61ru2P3mBw4qVUq7ZtDpelQDRrK9O8Zutm\n" \
"NHz6a4uPVymZ+DAXXbpyb/uBxa3Shlg9F8fnCbvxK/eG3MHacV3URuPMrSXBiLxg\n" \
"Z3Vms/EY96Jc5lP/Ooi2R6X/ExjqmAl3P51T+c8B5fWmcBcUr2Ok/5mzk53cU6cG\n" \
"/kiFHaFpriV1uxPMUgP17VGhi9sVAgMBAAGjggEIMIIBBDAOBgNVHQ8BAf8EBAMC\n" \
"AYYwHQYDVR0lBBYwFAYIKwYBBQUHAwIGCCsGAQUFBwMBMBIGA1UdEwEB/wQIMAYB\n" \
"Af8CAQAwHQYDVR0OBBYEFBQusxe3WFbLrlAJQOYfr52LFMLGMB8GA1UdIwQYMBaA\n" \
"FHm0WeZ7tuXkAXOACIjIGlj26ZtuMDIGCCsGAQUFBwEBBCYwJDAiBggrBgEFBQcw\n" \
"AoYWaHR0cDovL3gxLmkubGVuY3Iub3JnLzAnBgNVHR8EIDAeMBygGqAYhhZodHRw\n" \
"Oi8veDEuYy5sZW5jci5vcmcvMCIGA1UdIAQbMBkwCAYGZ4EMAQIBMA0GCysGAQQB\n" \
"gt8TAQEBMA0GCSqGSIb3DQEBCwUAA4ICAQCFyk5HPqP3hUSFvNVneLKYY611TR6W\n" \
"PTNlclQtgaDqw+34IL9fzLdwALduO/ZelN7kIJ+m74uyA+eitRY8kc607TkC53wl\n" \
"ikfmZW4/RvTZ8M6UK+5UzhK8jCdLuMGYL6KvzXGRSgi3yLgjewQtCPkIVz6D2QQz\n" \
"CkcheAmCJ8MqyJu5zlzyZMjAvnnAT45tRAxekrsu94sQ4egdRCnbWSDtY7kh+BIm\n" \
"lJNXoB1lBMEKIq4QDUOXoRgffuDghje1WrG9ML+Hbisq/yFOGwXD9RiX8F6sw6W4\n" \
"avAuvDszue5L3sz85K+EC4Y/wFVDNvZo4TYXao6Z0f+lQKc0t8DQYzk1OXVu8rp2\n" \
"yJMC6alLbBfODALZvYH7n7do1AZls4I9d1P4jnkDrQoxB3UqQ9hVl3LEKQ73xF1O\n" \
"yK5GhDDX8oVfGKF5u+decIsH4YaTw7mP3GFxJSqv3+0lUFJoi5Lc5da149p90Ids\n" \
"hCExroL1+7mryIkXPeFM5TgO9r0rvZaBFOvV2z0gp35Z0+L4WPlbuEjN/lxPFin+\n" \
"HlUjr8gRsI3qfJOQFy/9rKIJR0Y/8Omwt/8oTWgy1mdeHmmjk7j1nYsvC9JSQ6Zv\n" \
"MldlTTKB3zhThV1+XWYp6rjd5JW1zbVWEkLNxE7GJThEUG3szgBVGP7pSWTUTsqX\n" \
"nLRbwHOoq7hHwg==\n" \
"-----END CERTIFICATE-----\n";

const char *ssid = "fill-me-in";  // FYI The SSID can't have a space in it.
const char *password = "fill-me-in";

const i2s_port_t i2s_num = I2S_NUM_0; // i2s port number



WebsocketsClient webSocket;

const int buffer_size = 512;
int32_t i2s_read_buff_int32[512];
int16_t i2s_read_buff[512]; // Buffer for reading from microphone
size_t bytes_read;

unsigned long previousPing = 0;
const long pingInterval = 5000;


String generateRandomUniqueString() {
  // Seed the random number generator with an analog read for more randomness.
  // Use an unconnected analog pin to read noise.
  randomSeed(analogRead(0));

  // Generate a pseudo-random unique string.
  String uniqueString = "";
  for (int i = 0; i < 8; i++) { // Generate 8 characters for example
    char hexDigit = "0123456789ABCDEF"[random(0, 16)];
    uniqueString += hexDigit;
  }

  // Optionally, you can append microsecond count since boot to ensure uniqueness.
  uniqueString += String(micros(), HEX);

  return uniqueString;
}

// Function to play audio data from the queue.
void playAudioFromQueue() {
    if (!audioQueue.empty()) {
        // Get the front message (audio data) from the queue.
        std::vector<uint8_t> &audioData = audioQueue.front();

        // Initialize variables for chunked writing.
        size_t bytesWritten;

        i2s_write(I2S_NUM_0, audioData.data(), audioData.size(), &bytesWritten, portMAX_DELAY);

        // Remove the played audio data from the queue.
        audioQueue.pop();

        // If there's more data in the queue, continue playing.
        if (!audioQueue.empty()) {
            playAudioFromQueue();
        }
    }
}

void onMessageCallback(WebsocketsMessage message) {
    std::string rawData = message.rawData();

    const size_t numSamples = rawData.size() / sizeof(int16_t);
    std::vector<int32_t> scaledAudioData(numSamples);

    // This initial scaling factor moves int16 range to int32
    int32_t initialScalingFactor = 1 << 16;

    for(size_t i = 0; i < numSamples; i++) {
        int16_t sample16 = ((int16_t*)rawData.data())[i]; 
        // Convert to int32 and apply initial scaling to use the full range of int32
        int32_t sample32 = ((int32_t)sample16) * initialScalingFactor;
        scaledAudioData[i] = sample32;
    }

    size_t bytesWritten;
    esp_err_t err = i2s_write(I2S_NUM_0, scaledAudioData.data(), scaledAudioData.size() * sizeof(int32_t), &bytesWritten, portMAX_DELAY);
}

void onEventsCallback(WebsocketsEvent event, String data) {
    if(event == WebsocketsEvent::ConnectionOpened) {
        Serial.println("Connnection Opened");
    } else if(event == WebsocketsEvent::ConnectionClosed) {
        Serial.println("Connnection Closed");
    } else if(event == WebsocketsEvent::GotPing) {
        Serial.println("Got a Ping!");
    } else if(event == WebsocketsEvent::GotPong) {
        Serial.println("Got a Pong!");
    }
}

void connectToWifi(String ssid, String password) {
    WiFi.begin(ssid.c_str(), password.c_str());

    // Wait for connection to establish
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(1000);
        Serial.print(".");
        attempts++;
    }

    if(WiFi.status() == WL_CONNECTED) {
        Serial.println("Connected to Wi-Fi");
    } else {
        Serial.println("Failed to connect to Wi-Fi. Check credentials.");
    }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println("setting up wifi");
  WiFi.mode(WIFI_STA);
  connectToWifi(ssid, password);
  Serial.println("connected to wifi");
  Serial.println("let's try playing some audio");

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
    .sample_rate = 24000, // Adjust this according to your audio file's sample rate
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // Adjust this too
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // Mono
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0, // Default interrupt priority
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = true, // Auto clear tx descriptor on underflow
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = 16, // BCLK
    .ws_io_num = 9, // LRCK
    .data_out_num = 15, // Speaker output
    .data_in_num = 17 // Microphone input
  };

  // Install and start I2S driver with your new pin configuration
  i2s_driver_install(i2s_num, &i2s_config, 0, NULL);
  i2s_set_pin(i2s_num, &pin_config);  

  i2s_set_sample_rates(I2S_NUM_0, 22050); // Set the sample rate for your audio file.
  const char* audioURL = "http://quack.uberduck.ai/aec-cartoon.wav";
  // playAudioFromURL(audioURL);

  webSocket.setCACert(rootCACertificate);


  i2s_set_sample_rates(I2S_NUM_0, 16000);
  Serial.println("switched to 16khz sample rate");
  

  HTTPClient http;
  http.begin("https://9df3b80e5aae.ngrok.app/status");
  int httpCode = http.GET();
  if (httpCode == HTTP_CODE_OK) {
    auto stream = http.getStream();
    uint8_t buffer[1024];
    size_t bytesRead = 0;

    // Loop to read from the stream
    while ((bytesRead = stream.readBytes(buffer, sizeof(buffer))) > 0) {
      // Print each chunk of data as it's read
      Serial.write(buffer, bytesRead);
    }
    
  } else {
    Serial.printf("HTTP request failed: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
  Serial.println("connecting to websocket.");
  webSocket.onMessage(onMessageCallback);
  webSocket.onEvent(onEventsCallback);
  String session_id = generateRandomUniqueString();
  webSocket.connect(
    "wss://9df3b80e5aae.ngrok.app:443/audio/response?session_id=" + 
    session_id + 
    "&record=true" + 
    "&input_audio_format=int16&output_sample_rate=16000"
  );
  webSocket.ping();
}

void loop() {
  webSocket.poll();

  unsigned long currentMillis = millis();
  if (currentMillis - previousPing > pingInterval) {
    webSocket.ping();
    previousPing = currentMillis;
  }

  esp_err_t err = i2s_read(
      I2S_NUM_0,
      (void *) i2s_read_buff_int32,
      512 * sizeof(int32_t),
      &bytes_read,
      portMAX_DELAY
  );

  if (err == ESP_OK && bytes_read > 0) {
    for (int i = 0; i < bytes_read / sizeof(int32_t); i++) {
      // Simple right shift to convert int32 to int16, adjust this as needed
      i2s_read_buff[i] = (int16_t)(i2s_read_buff_int32[i] >> 16);
    }
    webSocket.sendBinary((const char*)i2s_read_buff, bytes_read / 2);
  }
}

void playAudioFromURL(const char* url) {
  HTTPClient http;
  http.begin(url);
  int httpCode = http.GET();
  if (httpCode == HTTP_CODE_OK) {
    auto stream = http.getStream();
    uint8_t buffer[512];
    size_t bytesRead = 0;
    size_t bytesWritten = 0;
    
    while ((bytesRead = stream.readBytes(buffer, sizeof(buffer))) > 0) {
      // Write data to I2S.
      i2s_write(I2S_NUM_0, buffer, bytesRead, &bytesWritten, portMAX_DELAY);
    }
  } else {
    Serial.printf("HTTP request failed: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}
