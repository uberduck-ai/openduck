// #include <esp_wifi.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "driver/i2s.h"

// #define BUFFER_SIZE     (1024)   // Size of the read buffer

const char *ssid = "fill-me-in";  // FYI The SSID can't have a space in it.
const char *password = "fill-me-in";

const i2s_port_t i2s_num = I2S_NUM_0; // i2s port number

const char* audioURL = "http://quack.uberduck.ai/aec-cartoon.wav";

// Buffer size and recording duration
#define SAMPLE_RATE 16000
#define RECORD_TIME_SECONDS 1
#define BUFFER_SIZE (SAMPLE_RATE * RECORD_TIME_SECONDS * 1) // *2 for 32 bits per sample

int32_t i2s_buffer[BUFFER_SIZE];


void record_audio() {
    size_t bytes_read;
    // Record audio
    i2s_read(I2S_NUM_0, &i2s_buffer, sizeof(i2s_buffer), &bytes_read, portMAX_DELAY);
}

void play_audio() {
    size_t bytes_written;
    // Play recorded audio
    i2s_write(I2S_NUM_0, &i2s_buffer, sizeof(i2s_buffer), &bytes_written, portMAX_DELAY);
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

  Serial.println("LittleFS mounted successfully");
  Serial.println("what is going on?");
  Serial.println("hi");
  Serial.println("setting up wifi");
  WiFi.mode(WIFI_STA);
  connectToWifi(ssid, password);
  Serial.println("connected to wifi");

  Serial.println("let's try playing some audio");

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
    .sample_rate = 44100, // Adjust this according to your audio file's sample rate
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // Adjust this too
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // Stereo
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0, // Default interrupt priority
    .dma_buf_count = 8,
    .dma_buf_len = 64,
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

  playAudioFromURL(audioURL);

  i2s_set_sample_rates(I2S_NUM_0, 16000);
  Serial.println("switched to 16khz sample rate");
      // Record audio
    Serial.println("Recording...");
    record_audio();
    Serial.println("Recording done.");

    // Delay to ensure recording has stopped before playback
    delay(1000);

    // Play recorded audio
    Serial.println("Playing back...");
    play_audio();
    Serial.println("Playback done.");

}

void loop() {
  // put your main code here, to run repeatedly:
    // Buffer to store I2S read data
    /*
  uint8_t i2s_read_buff[BUFFER_SIZE];
  size_t bytes_read;

  // Read a sample from the microphone
  esp_err_t err = i2s_read(I2S_NUM_0, &i2s_read_buff, BUFFER_SIZE, &bytes_read, portMAX_DELAY);


  // Play the sample back through the speaker
  size_t bytes_written;
  // Check for read success
  if (err == ESP_OK && bytes_read > 0) {
    // Print raw data or process as needed
    Serial.printf("Data read: %u bytes\n", bytes_read);
    // Example: Print the first byte of data
    Serial.printf("First byte: %u\n", i2s_read_buff[0]);
        for(int i = 0; i < bytes_read; i++) {
      // all 0 output
      Serial.printf("%d", (int)i2s_read_buff[i]);
    }

  } else {
    Serial.println("Error reading I2S data");
  }
  i2s_write(I2S_NUM_0, &i2s_read_buff, sizeof(i2s_read_buff), &bytes_written, portMAX_DELAY);

  delay(1000); // Delay to avoid overwhelming the serial output
  */
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

