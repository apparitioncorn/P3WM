/*
 * ==========================================================
 * P3 WALKMAN
 * Firmware Version: 0.3
 * Platform: ESP32 Dev Module
 * On Arduino IDE btw
 * ==========================================================
 *
 * What's new in v0.3:
 * - Bluetooth A2DP sink: pair your phone, stream Spotify to it
 * - Audio plays out of the internal DAC (GPIO25 = left, GPIO26 = right)
 * - AVRCP metadata (title, artist, duration) shown on the OLED
 * - Joystick = previous / next / play-pause, pot = volume
 * - The FFT now analyzes the REAL audio stream (44.1 kHz, 2048 pts)
 * - dB scaling + attack/release smoothing for the 7 bands
 * Wiring:
 * - OLED:     SDA -> GPIO21, SCL -> GPIO22, VCC -> 3.3V, GND -> GND
 * - Joystick: VRx -> GPIO34, SW -> GPIO14, VCC -> 3.3V, GND -> GND
 * - Pot:      middle pin -> GPIO36, outer pins -> 3.3V and GND
 * - Audio:    GPIO25 (left) / GPIO26 (right) -> headphone jack tip/ring,
 *             GND -> sleeve. Put ~100 ohm in series with each channel
 *             and keep volume modest. Direct drive is for testing only.
 *
 * Serial commands: help, status, info, memory, uptime, plot,
 *                  play, pause, next, prev
 * this is gonna be kinda difficult. let's see how it goes!
 */

#include <Arduino.h>
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "BluetoothA2DPSink.h"

// ==========================================================
// 1. configuration. will be changed later if joystick is not used. This will likely be a rotary encoder or some sort of tactile button, since a joystick just seems kind of clunky.
// ==========================================================

namespace Config {
  constexpr unsigned long SERIAL_BAUD = 115200;
  constexpr char DEVICE_NAME[] = "P3 Walkman";

  // --- Pins ---
  constexpr int PIN_SDA = 21;
  constexpr int PIN_SCL = 22;
  constexpr int PIN_JOY_X = 34;   // ADC1, input only
  constexpr int PIN_JOY_SW = 14;
  constexpr int PIN_VOLUME_POT = 36;  // ADC1, input only

  // --- OLED ---
  constexpr int OLED_WIDTH = 128;
  constexpr int OLED_HEIGHT = 64;
  constexpr uint8_t OLED_ADDRESS = 0x3C;  // try 0x3D if the screen stays blank
  constexpr unsigned long OLED_INTERVAL = 100;

  // --- audio visualizer ---
  constexpr int FFT_SIZE = 2048;           // ~21.5 Hz per bin at 44.1 kHz
  constexpr int RING_SIZE = 4096;          // must be a power of two
  constexpr int NUM_BANDS = 7;
  constexpr unsigned long VISUALIZER_INTERVAL = 33;  // ~30 fps
  constexpr float DB_FLOOR = -75.0f;       // reads as 0
  constexpr float DB_CEILING = -15.0f;     // reads as 100
  constexpr float ATTACK = 0.55f;          // how fast bars rise (0..1)
  constexpr float RELEASE = 0.12f;         // how fast bars fall (0..1)

  // --- controls ---
  constexpr int JOY_LOW = 800;
  constexpr int JOY_HIGH = 3300;
  constexpr int JOY_CENTER_MIN = 1500;
  constexpr int JOY_CENTER_MAX = 2600;
  constexpr unsigned long BUTTON_DEBOUNCE = 40;
  constexpr unsigned long AUDIO_TIMEOUT = 600;  // ms without samples = paused
}

constexpr float PI_F = 3.14159265358979323846f;

// ==========================================================
// 2. objects
// ==========================================================

BluetoothA2DPSink a2dpSink;
Adafruit_SSD1306 oled(Config::OLED_WIDTH, Config::OLED_HEIGHT, &Wire, -1);

struct SystemState {
  bool initialized = false;
  bool audioReady = false;
  bool displayReady = false;
  unsigned long uptime = 0;
  uint32_t freeHeap = 0;
};

SystemState systemState;
bool plotterMode = true;   // true = only plot data on serial, no text spaces
// ---- Audio ring buffer (written by the Bluetooth task) ----
int16_t ringBuffer[Config::RING_SIZE];
volatile uint32_t ringHead = 0;
volatile uint32_t lastAudioMs = 0;
volatile uint32_t sampleRate = 44100;

// ---- Track info (written by the Bluetooth task, read in loop) ----
portMUX_TYPE trackMux = portMUX_INITIALIZER_UNLOCKED;
char pendingTitle[64] = "";
char pendingArtist[64] = "";
volatile uint32_t pendingDurationMs = 0;
volatile bool trackInfoChanged = false;

char trackTitle[64] = "";
char trackArtist[64] = "";
uint32_t trackDurationMs = 0;
uint32_t elapsedMs = 0;
int scrollPixels = 0;

// ---- Visualizer ----
const char* bandNames[Config::NUM_BANDS] = {
  "Sub", "Bass", "LowMid", "Mid", "UpperMid", "Presence", "Treble"
};

// Band edges in Hz (NUM_BANDS + 1 values)
const float bandLimits[Config::NUM_BANDS + 1] = {
  20, 60, 250, 500, 2000, 4000, 8000, 16000
};

int bandFirstBin[Config::NUM_BANDS];
int bandLastBin[Config::NUM_BANDS];   // exclusive

float realData[Config::FFT_SIZE];
float imagData[Config::FFT_SIZE];
float hannWindow[Config::FFT_SIZE];
float windowSum = 0.0f;
float bandMagnitudes[Config::NUM_BANDS];   // smoothed, 0..100

// ---- Serial input ----
constexpr size_t COMMAND_BUFFER_SIZE = 48;
char commandBuffer[COMMAND_BUFFER_SIZE];
size_t commandLength = 0;
bool discardingLine = false;

// ==========================================================
// 3. FUNCTION DECLARATIONS
// ==========================================================

void initializeSystem();
void initializeAudio();
void initializeDisplays();
void initializeControls();
void initializeVisualizer();

void updateSystem();
void updateDiagnostics();
void processSerialCommands();
void executeSerialCommand(const char* command);
void printSystemInformation();
void printHelp();

void updateControls();
void updateTrackInfo();
void updateDisplay();
bool isAudioActive();

void updateVisualizer();
void recomputeBandBins();
void loadFromRingBuffer();
void performFFT();
void calculateMagnitudes();
void extractFrequencyBands();
void displaySpectrum();
void reverseBits();

// ==========================================================
// 4. BLUETOOTH CALLBACKS (run on the Bluetooth task)
// ==========================================================

void onAudioData(const uint8_t* data, uint32_t length) {
  const int16_t* samples = reinterpret_cast<const int16_t*>(data);
  const uint32_t frames = length / 4;   // 2 channels x 2 bytes

  uint32_t head = ringHead;
  for (uint32_t i = 0; i < frames; i++) {
    const int32_t mono =
      (static_cast<int32_t>(samples[2 * i]) + samples[2 * i + 1]) / 2;
    ringBuffer[head & (Config::RING_SIZE - 1)] = static_cast<int16_t>(mono);
    head++;
  }
  ringHead = head;
  lastAudioMs = millis();
}

void onSampleRate(uint16_t rate) {
  sampleRate = rate;
}

// Copies text, replaces anything the OLED cannot draw on it's own i think
static void copyAscii(char* destination, size_t size, const uint8_t* text) {
  size_t i = 0;
  for (; i < size - 1 && text[i] != '\0'; i++) {
    destination[i] = (text[i] >= 32 && text[i] < 127)
                       ? static_cast<char>(text[i]) : '?';
  }
  destination[i] = '\0';
}

void onMetadata(uint8_t attributeId, const uint8_t* text) {
  portENTER_CRITICAL(&trackMux);

  if (attributeId == ESP_AVRC_MD_ATTR_TITLE) {
    copyAscii(pendingTitle, sizeof(pendingTitle), text);
    trackInfoChanged = true;
  }
  else if (attributeId == ESP_AVRC_MD_ATTR_ARTIST) {
    copyAscii(pendingArtist, sizeof(pendingArtist), text);
    trackInfoChanged = true;
  }
  else if (attributeId == ESP_AVRC_MD_ATTR_PLAYING_TIME) {
    pendingDurationMs = strtoul(reinterpret_cast<const char*>(text),
                                nullptr, 10);
    trackInfoChanged = true;
  }

  portEXIT_CRITICAL(&trackMux);
}

// ==========================================================
// 5. SETUP AND LOOP
// ==========================================================

void setup() {
  Serial.begin(Config::SERIAL_BAUD);
  delay(500);

  Serial.println();
  Serial.println("================================");
  Serial.println("       P3 WALKMAN");
  Serial.println("       FIRMWARE v0.3");
  Serial.println("================================");

  initializeSystem();
}

void loop() {
  updateSystem();
  processSerialCommands();
  updateDiagnostics();
  updateControls();
  updateTrackInfo();
  updateVisualizer();
  updateDisplay();
}

// ==========================================================
// 6. INITIALIZATION
// ==========================================================

void initializeSystem() {
  Serial.println("[SYSTEM] Initializing...");

  initializeVisualizer();
  initializeDisplays();
  initializeControls();
  initializeAudio();   // last, so the screen can show a message first

  systemState.initialized = true;
  Serial.println("[SYSTEM] Ready.");
  printHelp();
}

void initializeVisualizer() {
  windowSum = 0.0f;
  for (int i = 0; i < Config::FFT_SIZE; i++) {
    hannWindow[i] = 0.5f * (1.0f - cosf(
      2.0f * PI_F * static_cast<float>(i) /
      static_cast<float>(Config::FFT_SIZE - 1)));
    windowSum += hannWindow[i];
  }

  for (int i = 0; i < Config::NUM_BANDS; i++) {
    bandMagnitudes[i] = 0.0f;
  }

  recomputeBandBins();
}

// Maps band edges (Hz) to FFT bins for the current sample rate.
void recomputeBandBins() {
  const float binWidth =
    static_cast<float>(sampleRate) / Config::FFT_SIZE;

  for (int band = 0; band < Config::NUM_BANDS; band++) {
    int first = static_cast<int>(lroundf(bandLimits[band] / binWidth));
    int last = static_cast<int>(lroundf(bandLimits[band + 1] / binWidth));

    if (first < 1) first = 1;                 
    if (last <= first) last = first + 1;      // one bin
    if (last > Config::FFT_SIZE / 2) last = Config::FFT_SIZE / 2;

    bandFirstBin[band] = first;
    bandLastBin[band] = last;               // exclusive
  }
}

void initializeDisplays() {
  Wire.begin(Config::PIN_SDA, Config::PIN_SCL);
  Wire.setClock(400000);

  if (!oled.begin(SSD1306_SWITCHCAPVCC, Config::OLED_ADDRESS)) {
    Serial.println("[DISPLAY] OLED not found. Check wiring / address.");
    systemState.displayReady = false;
    return;
  }

  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println("P3 WALKMAN v0.3");
  oled.println("Starting Bluetooth...");
  oled.display();

  systemState.displayReady = true;
  Serial.println("[DISPLAY] OLED ready.");
}

void initializeControls() {
  pinMode(Config::PIN_JOY_SW, INPUT_PULLUP);
  pinMode(Config::PIN_JOY_X, INPUT);
  pinMode(Config::PIN_VOLUME_POT, INPUT);
  Serial.println("[CONTROLS] Joystick and volume pot ready.");
}

void initializeAudio() {
  // Default output is the internal DAC on GPIO25 / GPIO26.
  a2dpSink.set_raw_stream_reader(onAudioData);
  a2dpSink.set_sample_rate_callback(onSampleRate);
  a2dpSink.set_avrc_metadata_attribute_mask(
    ESP_AVRC_MD_ATTR_TITLE |
    ESP_AVRC_MD_ATTR_ARTIST |
    ESP_AVRC_MD_ATTR_PLAYING_TIME);
  a2dpSink.set_avrc_metadata_callback(onMetadata);
  a2dpSink.set_auto_reconnect(true);
  a2dpSink.start(Config::DEVICE_NAME);

  systemState.audioReady = true;
  Serial.print("[AUDIO] Bluetooth started. Pair with: ");
  Serial.println(Config::DEVICE_NAME);
}

// ==========================================================
// 7. SYSTEM UPDATE AND DIAGNOSTICS
// ==========================================================

void updateSystem() {
  static unsigned long previousUpdate = 0;
  const unsigned long now = millis();

  if (now - previousUpdate >= 1000) {
    previousUpdate = now;
    systemState.uptime = now / 1000;
    systemState.freeHeap = ESP.getFreeHeap();
  }
}

void updateDiagnostics() {
  static unsigned long previousDiagnostic = 0;
  const unsigned long now = millis();

  if (now - previousDiagnostic >= 5000) {
    previousDiagnostic = now;

    if (!plotterMode) {
      Serial.print("[DIAGNOSTICS] Uptime: ");
      Serial.print(systemState.uptime);
      Serial.print(" s | Free heap: ");
      Serial.print(systemState.freeHeap);
      Serial.println(" bytes");
    }
  }
}

// ==========================================================
// 8. CONTROLS
// ==========================================================

bool isAudioActive() {
  return (millis() - lastAudioMs) < Config::AUDIO_TIMEOUT;
}

void updateControls() {
  // ---- Joystick left/right = previous / next ----
  static bool joystickArmed = true;
  static float joystickX = 2048.0f;

  joystickX = joystickX * 0.7f + analogRead(Config::PIN_JOY_X) * 0.3f;
  const int x = static_cast<int>(joystickX);

  if (x > Config::JOY_CENTER_MIN && x < Config::JOY_CENTER_MAX) {
    joystickArmed = true;   // must return to center before the next trigger
  }
  else if (joystickArmed && a2dpSink.is_connected()) {
    if (x < Config::JOY_LOW) {
      a2dpSink.previous();
      joystickArmed = false;
    }
    else if (x > Config::JOY_HIGH) {
      a2dpSink.next();
      joystickArmed = false;
    }
  }

  // ---- Joystick press = play / pause ----
  static bool lastButton = HIGH;
  static unsigned long lastButtonChange = 0;
  const bool button = digitalRead(Config::PIN_JOY_SW);
  const unsigned long now = millis();

  if (button != lastButton && now - lastButtonChange > Config::BUTTON_DEBOUNCE) {
    lastButtonChange = now;
    lastButton = button;

    if (button == LOW && a2dpSink.is_connected()) {
      if (isAudioActive()) {
        a2dpSink.pause();
      } else {
        a2dpSink.play();
      }
    }
  }

  // ---- Pot = volume (0..127) ----
  static float potSmooth = 0.0f;
  static int lastVolume = -10;

  potSmooth = potSmooth * 0.9f + analogRead(Config::PIN_VOLUME_POT) * 0.1f;
  const int volume = static_cast<int>(potSmooth * 127.0f / 4095.0f);

  if (abs(volume - lastVolume) >= 2 && a2dpSink.is_connected()) {
    lastVolume = volume;
    a2dpSink.set_volume(static_cast<uint8_t>(volume));
  }
}

// ==========================================================
// 9. TRACK INFO AND OLED
// ==========================================================

void updateTrackInfo() {
  if (!trackInfoChanged) {
    return;
  }

  char title[64];
  char artist[64];
  uint32_t duration;

  portENTER_CRITICAL(&trackMux);
  strncpy(title, pendingTitle, sizeof(title));
  strncpy(artist, pendingArtist, sizeof(artist));
  title[sizeof(title) - 1] = '\0';
  artist[sizeof(artist) - 1] = '\0';
  duration = pendingDurationMs;
  trackInfoChanged = false;
  portEXIT_CRITICAL(&trackMux);

  if (strcmp(title, trackTitle) != 0) {
    // New track: restart the timer and the scrolling.
    elapsedMs = 0;
    scrollPixels = 0;
  }

  strncpy(trackTitle, title, sizeof(trackTitle));
  strncpy(trackArtist, artist, sizeof(trackArtist));
  trackDurationMs = duration;

  if (!plotterMode) {
    Serial.print("[TRACK] ");
    Serial.print(trackTitle);
    Serial.print(" - ");
    Serial.print(trackArtist);
    Serial.print(" (");
    Serial.print(trackDurationMs / 1000);
    Serial.println(" s)");
  }
}

static void formatTime(char* out, size_t size, uint32_t ms) {
  const uint32_t totalSeconds = ms / 1000;
  snprintf(out, size, "%lu:%02lu",
           static_cast<unsigned long>(totalSeconds / 60),
           static_cast<unsigned long>(totalSeconds % 60));
}

// Draws text that scrolls horizontally if it is wider than the screen.
static void drawMarquee(const char* text, int y, int textSize) {
  const int charWidth = 6 * textSize;
  const int textWidth = static_cast<int>(strlen(text)) * charWidth;

  oled.setTextSize(textSize);
  oled.setTextWrap(false);

  if (textWidth <= Config::OLED_WIDTH) {
    oled.setCursor(0, y);
    oled.print(text);
    return;
  }

  const int gap = 40;
  const int cycle = textWidth + gap;
  const int offset = scrollPixels % cycle;

  oled.setCursor(-offset, y);
  oled.print(text);
  oled.setCursor(-offset + cycle, y);
  oled.print(text);
}

void updateDisplay() {
  if (!systemState.displayReady) {
    return;
  }

  static unsigned long lastDraw = 0;
  const unsigned long now = millis();

  if (now - lastDraw < Config::OLED_INTERVAL) {
    return;
  }

  const unsigned long dt = now - lastDraw;
  lastDraw = now;

  const bool connected = a2dpSink.is_connected();
  const bool playing = isAudioActive();

  if (playing) {
    elapsedMs += dt;
    scrollPixels += 3;
  }

  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);

  // Status line
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  if (!connected) {
    oled.print("Pair: ");
    oled.print(Config::DEVICE_NAME);
  } else {
    oled.print(playing ? "> PLAYING" : "|| PAUSED");
  }

  if (connected && trackTitle[0] != '\0') {
    drawMarquee(trackTitle, 14, 2);
    drawMarquee(trackArtist, 36, 1);
  } else if (connected) {
    oled.setCursor(0, 20);
    oled.print("Waiting for track...");
  }

  // Progress bar and time
  if (connected) {
    const int barY = 50;
    oled.drawRect(0, barY, Config::OLED_WIDTH, 5, SSD1306_WHITE);

    if (trackDurationMs > 0) {
      uint32_t shown = elapsedMs > trackDurationMs ? trackDurationMs : elapsedMs;
      const int fill = static_cast<int>(
        (Config::OLED_WIDTH - 4) * static_cast<float>(shown) / trackDurationMs);
      oled.fillRect(2, barY + 2, fill, 1, SSD1306_WHITE);
    }

    char elapsedText[12];
    char durationText[12];
    formatTime(elapsedText, sizeof(elapsedText), elapsedMs);

    oled.setTextSize(1);
    oled.setCursor(0, 56);
    oled.print(elapsedText);

    if (trackDurationMs > 0) {
      formatTime(durationText, sizeof(durationText), trackDurationMs);
    } else {
      strcpy(durationText, "--:--");
    }

    const int durationWidth = static_cast<int>(strlen(durationText)) * 6;
    oled.setCursor(Config::OLED_WIDTH - durationWidth, 56);
    oled.print(durationText);
  }

  oled.display();
}

// ==========================================================
// 10. VISUALIZER
// ==========================================================

void updateVisualizer() {
  static unsigned long lastUpdate = 0;
  static uint32_t lastRate = 0;
  const unsigned long now = millis();

  if (now - lastUpdate < Config::VISUALIZER_INTERVAL) {
    return;
  }
  lastUpdate = now;

  // The phone can change sample rate (44.1k / 48k): rebuild the band map.
  if (sampleRate != lastRate) {
    lastRate = sampleRate;
    recomputeBandBins();
  }

  if (isAudioActive()) {
    loadFromRingBuffer();
    performFFT();
    calculateMagnitudes();
    extractFrequencyBands();   // updates bandMagnitudes with smoothing
  } else {
    // No audio: let the bars fall to zero.
    for (int i = 0; i < Config::NUM_BANDS; i++) {
      bandMagnitudes[i] *= (1.0f - Config::RELEASE);
    }
  }

  displaySpectrum();
}

// Takes the newest FFT_SIZE samples, windows them, loads the FFT buffers.
void loadFromRingBuffer() {
  const uint32_t head = ringHead;

  for (int i = 0; i < Config::FFT_SIZE; i++) {
    const uint32_t index =
      (head - Config::FFT_SIZE + i) & (Config::RING_SIZE - 1);
    realData[i] = (ringBuffer[index] / 32768.0f) * hannWindow[i];
    imagData[i] = 0.0f;
  }
}

void reverseBits() {
  int j = 0;

  for (int i = 1; i < Config::FFT_SIZE; i++) {
    int bit = Config::FFT_SIZE >> 1;

    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;

    if (i < j) {
      const float tempReal = realData[i];
      realData[i] = realData[j];
      realData[j] = tempReal;

      const float tempImag = imagData[i];
      imagData[i] = imagData[j];
      imagData[j] = tempImag;
    }
  }
}

void performFFT() {
  reverseBits();

  for (int size = 2; size <= Config::FFT_SIZE; size *= 2) {
    const int halfSize = size / 2;
    const float angle = -2.0f * PI_F / static_cast<float>(size);
    const float wRealStep = cosf(angle);
    const float wImagStep = sinf(angle);

    for (int start = 0; start < Config::FFT_SIZE; start += size) {
      float wReal = 1.0f;
      float wImag = 0.0f;

      for (int j = 0; j < halfSize; j++) {
        const int evenIndex = start + j;
        const int oddIndex = evenIndex + halfSize;

        const float oddReal =
          realData[oddIndex] * wReal - imagData[oddIndex] * wImag;
        const float oddImag =
          realData[oddIndex] * wImag + imagData[oddIndex] * wReal;

        const float evenReal = realData[evenIndex];
        const float evenImag = imagData[evenIndex];

        realData[evenIndex] = evenReal + oddReal;
        imagData[evenIndex] = evenImag + oddImag;
        realData[oddIndex] = evenReal - oddReal;
        imagData[oddIndex] = evenImag - oddImag;

        const float nextWReal = wReal * wRealStep - wImag * wImagStep;
        const float nextWImag = wReal * wImagStep + wImag * wRealStep;
        wReal = nextWReal;
        wImag = nextWImag;
      }
    }
  }
}

void calculateMagnitudes() {
  const float normalization = 2.0f / windowSum;

  for (int i = 0; i < Config::FFT_SIZE / 2; i++) {
    const float magnitude = sqrtf(
      realData[i] * realData[i] + imagData[i] * imagData[i]);
    realData[i] = magnitude * normalization;
  }
}

void extractFrequencyBands() {
  for (int band = 0; band < Config::NUM_BANDS; band++) {
    float peak = 0.0f;

    for (int bin = bandFirstBin[band]; bin < bandLastBin[band]; bin++) {
      if (realData[bin] > peak) {
        peak = realData[bin];
      }
    }

    // Amplitude -> dB -> 0..100
    const float decibels = 20.0f * log10f(peak + 1e-7f);
    float target = (decibels - Config::DB_FLOOR) /
                   (Config::DB_CEILING - Config::DB_FLOOR) * 100.0f;
    if (target < 0.0f) target = 0.0f;
    if (target > 100.0f) target = 100.0f;

    // Fast attack, slow release
    const float rate =
      (target > bandMagnitudes[band]) ? Config::ATTACK : Config::RELEASE;
    bandMagnitudes[band] += (target - bandMagnitudes[band]) * rate;
  }
}

void displaySpectrum() {
  if (!plotterMode) {
    return;
  }

  for (int i = 0; i < Config::NUM_BANDS; i++) {
    Serial.print(bandNames[i]);
    Serial.print(":");
    Serial.print(bandMagnitudes[i], 1);

    if (i < Config::NUM_BANDS - 1) {
      Serial.print('\t');
    }
  }
  Serial.println();
}

// ==========================================================
// 11. SERIAL COMMANDS
// ==========================================================

void processSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      if (discardingLine) {
        discardingLine = false;
      } else if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        executeSerialCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }

    if (discardingLine) {
      continue;
    }

    if (commandLength < COMMAND_BUFFER_SIZE - 1) {
      commandBuffer[commandLength++] =
        (incoming >= 'A' && incoming <= 'Z') ? incoming - 'A' + 'a' : incoming;
    } else {
      commandLength = 0;
      discardingLine = true;
      Serial.println("[ERROR] Command too long.");
    }
  }
}

void executeSerialCommand(const char* command) {
  if (strcmp(command, "help") == 0) {
    printHelp();
  }
  else if (strcmp(command, "status") == 0) {
    printSystemInformation();
  }
  else if (strcmp(command, "info") == 0) {
    Serial.println("[SYSTEM] P3 Walkman Firmware v0.3");
    Serial.print("[SYSTEM] Chip: ");
    Serial.println(ESP.getChipModel());
    Serial.print("[SYSTEM] CPU: ");
    Serial.print(ESP.getCpuFreqMHz());
    Serial.println(" MHz");
  }
  else if (strcmp(command, "memory") == 0) {
    Serial.print("[MEMORY] Free heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
    Serial.print("[MEMORY] Minimum free heap: ");
    Serial.print(ESP.getMinFreeHeap());
    Serial.println(" bytes");
  }
  else if (strcmp(command, "uptime") == 0) {
    Serial.print("[SYSTEM] Uptime: ");
    Serial.print(millis() / 1000);
    Serial.println(" seconds");
  }
  else if (strcmp(command, "plot") == 0) {
    plotterMode = !plotterMode;
    Serial.print("[SYSTEM] Plotter mode ");
    Serial.println(plotterMode ? "ON" : "OFF");
  }
  else if (strcmp(command, "play") == 0) {
    a2dpSink.play();
  }
  else if (strcmp(command, "pause") == 0) {
    a2dpSink.pause();
  }
  else if (strcmp(command, "next") == 0) {
    a2dpSink.next();
  }
  else if (strcmp(command, "prev") == 0) {
    a2dpSink.previous();
  }
  else {
    Serial.print("[ERROR] Unknown command: ");
    Serial.println(command);
    Serial.println("Type 'help' for commands.");
  }
}

void printSystemInformation() {
  Serial.println();
  Serial.println("========== SYSTEM STATUS ==========");
  Serial.println("Firmware: P3 Walkman v0.3");

  Serial.print("Bluetooth: ");
  Serial.println(a2dpSink.is_connected() ? "CONNECTED" : "WAITING FOR PHONE");

  Serial.print("Audio stream: ");
  Serial.println(isAudioActive() ? "ACTIVE" : "IDLE");

  Serial.print("Sample rate: ");
  Serial.print(sampleRate);
  Serial.println(" Hz");

  Serial.print("Display: ");
  Serial.println(systemState.displayReady ? "READY" : "NOT FOUND");

  Serial.print("Track: ");
  Serial.print(trackTitle);
  Serial.print(" - ");
  Serial.println(trackArtist);

  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println(" seconds");

  Serial.print("Free Heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");
  Serial.println("===================================");
  Serial.println();
}

void printHelp() {
  Serial.println();
  Serial.println("========== COMMANDS ==========");
  Serial.println("help   - Display commands");
  Serial.println("status - System status");
  Serial.println("info   - ESP32 information");
  Serial.println("memory - Memory diagnostics");
  Serial.println("uptime - System uptime");
  Serial.println("plot   - Toggle plotter-only output");
  Serial.println("play / pause / next / prev");
  Serial.println("==============================");
  Serial.println();
}