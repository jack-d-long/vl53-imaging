#include <Arduino.h>
#include <Wire.h>
#include <string.h>
#include "SparkFun_VL53L1X.h"

// ----------------------------
// User configuration
// ----------------------------

  #define SERIAL_BAUD 115200
  const uint8_t ROI_STEP_X = 1;
  const uint8_t ROI_STEP_Y = 1;
  const uint8_t MAX_ZONE_ATTEMPTS = 1;

// ESP32 users can set explicit SDA/SCL pins here. Other boards ignore these when USE_CUSTOM_I2C_PINS is false.
#define USE_CUSTOM_I2C_PINS false
const int I2C_SDA_PIN = 21;
const int I2C_SCL_PIN = 22;
const uint32_t I2C_CLOCK_HZ = 400000;

// Default VL53L1X address is 0x29 after boot. For a single-sensor sketch, leave this unchanged.
const uint8_t I2C_ADDRESS = 0x29;

// Optional pins for boards that wire XSHUT / GPIO1. Leave negative if unused.
const int SHUTDOWN_PIN = -1;
const int INTERRUPT_PIN = -1;
#define USE_SENSOR_CONTROL_PINS false

// Minimum ROI is 4x4. This sketch defaults to the maximum spatial sampling density.
const uint8_t ROI_WIDTH = 4;
const uint8_t ROI_HEIGHT = 4;

// Clip outermost ROI placements if edge cells are unstable.
// Example: set both to 1 to shrink a 13x13 sweep to 11x11.
const uint8_t EDGE_CLIP_X = 0;
const uint8_t EDGE_CLIP_Y = 0;

#define SENSOR_MODE_INTENSITY 1
#define SENSOR_MODE_DEPTH 2
const uint8_t SENSOR_MODE = SENSOR_MODE_INTENSITY;

// SparkFun exposes valid budgets of 15, 20, 33, 50, 100, 200, 500 ms.
const uint16_t TIMING_BUDGET_MS = (SENSOR_MODE == SENSOR_MODE_INTENSITY) ? 20 : 50;

// UM2555 recommends programming the next ROI center before the next range starts.
// Keep intermeasurement slightly larger than the timing budget to leave a reconfiguration gap.
const uint16_t INTER_MEASUREMENT_MS = TIMING_BUDGET_MS + 2;

// When true, range-status values other than 0 remain in the JSON stream and the visualizer masks them.
const bool REPORT_INVALID_ZONES = true;

// Retry the same ROI this many times if range status is non-zero.
// This improves fill rate at the cost of frame time.

// ----------------------------
// Verified SPAD map
// ----------------------------
// Source: ST UM2555 Rev 3, section 3.1 "SPAD locations".
// Orientation matches the printed ST table: rows top->bottom, columns left->right, viewed from behind the device
// looking toward the target, with Pin 1 at the upper-left of the table.
static const uint8_t SPAD_MAP[16][16] = {
  {128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248},
  {129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249},
  {130, 138, 146, 154, 162, 170, 178, 186, 194, 202, 210, 218, 226, 234, 242, 250},
  {131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251},
  {132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252},
  {133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253},
  {134, 142, 150, 158, 166, 174, 182, 190, 198, 206, 214, 222, 230, 238, 246, 254},
  {135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255},
  {127, 119, 111, 103,  95,  87,  79,  71,  63,  55,  47,  39,  31,  23,  15,   7},
  {126, 118, 110, 102,  94,  86,  78,  70,  62,  54,  46,  38,  30,  22,  14,   6},
  {125, 117, 109, 101,  93,  85,  77,  69,  61,  53,  45,  37,  29,  21,  13,   5},
  {124, 116, 108, 100,  92,  84,  76,  68,  60,  52,  44,  36,  28,  20,  12,   4},
  {123, 115, 107,  99,  91,  83,  75,  67,  59,  51,  43,  35,  27,  19,  11,   3},
  {122, 114, 106,  98,  90,  82,  74,  66,  58,  50,  42,  34,  26,  18,  10,   2},
  {121, 113, 105,  97,  89,  81,  73,  65,  57,  49,  41,  33,  25,  17,   9,   1},
  {120, 112, 104,  96,  88,  80,  72,  64,  56,  48,  40,  32,  24,  16,   8,   0},
};

static_assert(ROI_WIDTH >= 4 && ROI_WIDTH <= 16, "ROI_WIDTH must be within 4..16");
static_assert(ROI_HEIGHT >= 4 && ROI_HEIGHT <= 16, "ROI_HEIGHT must be within 4..16");
static_assert(ROI_STEP_X >= 1 && ROI_STEP_Y >= 1, "ROI step must be >= 1");

constexpr uint8_t GRID_COLS = ((16 - ROI_WIDTH) / ROI_STEP_X) + 1;
constexpr uint8_t GRID_ROWS = ((16 - ROI_HEIGHT) / ROI_STEP_Y) + 1;
constexpr uint8_t ACTIVE_GRID_COLS = GRID_COLS - (2 * EDGE_CLIP_X);
constexpr uint8_t ACTIVE_GRID_ROWS = GRID_ROWS - (2 * EDGE_CLIP_Y);
constexpr uint16_t ZONE_COUNT = ACTIVE_GRID_ROWS * ACTIVE_GRID_COLS;

static_assert((2 * EDGE_CLIP_X) < GRID_COLS, "EDGE_CLIP_X too large for current ROI/stride");
static_assert((2 * EDGE_CLIP_Y) < GRID_ROWS, "EDGE_CLIP_Y too large for current ROI/stride");

#if USE_SENSOR_CONTROL_PINS
SFEVL53L1X distanceSensor(Wire, SHUTDOWN_PIN, INTERRUPT_PIN);
#else
SFEVL53L1X distanceSensor;
#endif

uint32_t frameCounter = 0;
char commandBuffer[32];
uint8_t commandLength = 0;

uint8_t centerRowFromTop(uint8_t topRow) {
  return topRow + ((ROI_HEIGHT - 1) / 2);
}

uint8_t centerColFromLeft(uint8_t leftCol) {
  return leftCol + (ROI_WIDTH / 2);
}

uint8_t centerSpadForZone(uint8_t zoneRow, uint8_t zoneCol) {
  const uint8_t topRow = (zoneRow + EDGE_CLIP_Y) * ROI_STEP_Y;
  const uint8_t leftCol = (zoneCol + EDGE_CLIP_X) * ROI_STEP_X;
  const uint8_t centerRow = centerRowFromTop(topRow);
  const uint8_t centerCol = centerColFromLeft(leftCol);
  return SPAD_MAP[centerRow][centerCol];
}

struct ZoneMeasurement {
  uint16_t signalRateKcps;
  uint16_t ambientRateKcps;
  uint16_t distanceMm;
  uint8_t rangeStatus;
  uint8_t attempts;
};

void beginWire() {
#if defined(ARDUINO_ARCH_ESP32)
  if (USE_CUSTOM_I2C_PINS) {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  } else {
    Wire.begin();
  }
#else
  Wire.begin();
#endif
  Wire.setClock(I2C_CLOCK_HZ);
}

[[noreturn]] void fatal(const __FlashStringHelper *message) {
  while (true) {
    Serial.print(F("{\"type\":\"error\",\"message\":\""));
    Serial.print(message);
    Serial.println(F("\"}"));
    delay(1000);
  }
}

bool waitForDataReady(uint32_t timeoutMs) {
  const uint32_t startMs = millis();
  while (!distanceSensor.checkForDataReady()) {
    if ((millis() - startMs) > timeoutMs) {
      return false;
    }
    delay(1);
  }
  return true;
}

void configureSensor() {
  distanceSensor.init();

  if (I2C_ADDRESS != 0x29) {
    distanceSensor.setI2CAddress(I2C_ADDRESS);
  }

  if (!distanceSensor.checkBootState()) {
    fatal(F("VL53L1X boot check failed"));
  }

  if (!distanceSensor.checkID()) {
    fatal(F("Unexpected sensor ID"));
  }

  if (SENSOR_MODE == SENSOR_MODE_INTENSITY) {
    distanceSensor.setDistanceModeShort();
  } else {
    distanceSensor.setDistanceModeLong();
  }

  distanceSensor.setTimingBudgetInMs(TIMING_BUDGET_MS);
  distanceSensor.setIntermeasurementPeriod(INTER_MEASUREMENT_MS);
  distanceSensor.setROI(ROI_WIDTH, ROI_HEIGHT, centerSpadForZone(0, 0));
}

void printState(const __FlashStringHelper *state) {
  Serial.print(F("{\"type\":\"state\",\"state\":\""));
  Serial.print(state);
  Serial.println(F("\"}"));
}

void printMeta() {
  Serial.print(F("{\"type\":\"meta\""));
  Serial.print(F(",\"sensor\":\"VL53L1X\""));
  Serial.print(F(",\"library\":\"SparkFun_VL53L1X\""));
  Serial.print(F(",\"roi_width\":"));
  Serial.print(ROI_WIDTH);
  Serial.print(F(",\"roi_height\":"));
  Serial.print(ROI_HEIGHT);
  Serial.print(F(",\"step_x\":"));
  Serial.print(ROI_STEP_X);
  Serial.print(F(",\"step_y\":"));
  Serial.print(ROI_STEP_Y);
  Serial.print(F(",\"rows\":"));
  Serial.print(ACTIVE_GRID_ROWS);
  Serial.print(F(",\"cols\":"));
  Serial.print(ACTIVE_GRID_COLS);
  Serial.print(F(",\"zone_count\":"));
  Serial.print(ZONE_COUNT);
  Serial.print(F(",\"timing_budget_ms\":"));
  Serial.print(TIMING_BUDGET_MS);
  Serial.print(F(",\"inter_measurement_ms\":"));
  Serial.print(INTER_MEASUREMENT_MS);
  Serial.print(F(",\"mode\":\""));
  Serial.print((SENSOR_MODE == SENSOR_MODE_INTENSITY) ? F("intensity") : F("depth"));
  Serial.print(F("\""));
  Serial.print(F(",\"scan_order\":\"row-major\""));
  Serial.print(F(",\"centers\":["));

  bool first = true;
  for (uint8_t zoneRow = 0; zoneRow < ACTIVE_GRID_ROWS; ++zoneRow) {
    for (uint8_t zoneCol = 0; zoneCol < ACTIVE_GRID_COLS; ++zoneCol) {
      if (!first) {
        Serial.print(',');
      }
      first = false;
      const uint8_t topRow = (zoneRow + EDGE_CLIP_Y) * ROI_STEP_Y;
      const uint8_t leftCol = (zoneCol + EDGE_CLIP_X) * ROI_STEP_X;
      const uint8_t centerRow = centerRowFromTop(topRow);
      const uint8_t centerCol = centerColFromLeft(leftCol);
      const uint8_t centerSpad = SPAD_MAP[centerRow][centerCol];

      Serial.print(F("{\"zone\":"));
      Serial.print(static_cast<uint16_t>(zoneRow) * ACTIVE_GRID_COLS + zoneCol);
      Serial.print(F(",\"grid_row\":"));
      Serial.print(zoneRow);
      Serial.print(F(",\"grid_col\":"));
      Serial.print(zoneCol);
      Serial.print(F(",\"spad_center\":"));
      Serial.print(centerSpad);
      Serial.print(F(",\"center_row\":"));
      Serial.print(centerRow);
      Serial.print(F(",\"center_col\":"));
      Serial.print(centerCol);
      Serial.print(F(",\"roi_top\":"));
      Serial.print(topRow);
      Serial.print(F(",\"roi_left\":"));
      Serial.print(leftCol);
      Serial.print(F("}"));
    }
  }

  Serial.println(F("]}"));
}

void printFrame() {
  const uint32_t frameStartMs = millis();
  const uint32_t readyTimeoutMs = INTER_MEASUREMENT_MS + TIMING_BUDGET_MS + 25;

  distanceSensor.setROI(ROI_WIDTH, ROI_HEIGHT, centerSpadForZone(0, 0));
  distanceSensor.startRanging();

  Serial.print(F("{\"type\":\"frame\""));
  Serial.print(F(",\"frame\":"));
  Serial.print(frameCounter++);
  Serial.print(F(",\"rows\":"));
  Serial.print(ACTIVE_GRID_ROWS);
  Serial.print(F(",\"cols\":"));
  Serial.print(ACTIVE_GRID_COLS);
  Serial.print(F(",\"zone_count\":"));
  Serial.print(ZONE_COUNT);
  Serial.print(F(",\"zones\":["));

  bool firstZone = true;
  for (uint8_t zoneRow = 0; zoneRow < ACTIVE_GRID_ROWS; ++zoneRow) {
    for (uint8_t zoneCol = 0; zoneCol < ACTIVE_GRID_COLS; ++zoneCol) {
      const uint16_t zoneIndex = static_cast<uint16_t>(zoneRow) * ACTIVE_GRID_COLS + zoneCol;
      ZoneMeasurement measurement = {0, 0, 0, 255, 0};

      for (uint8_t attempt = 0; attempt < MAX_ZONE_ATTEMPTS; ++attempt) {
        if (!waitForDataReady(readyTimeoutMs)) {
          distanceSensor.stopRanging();
          fatal(F("Timeout waiting for ranging data"));
        }

        measurement.signalRateKcps = distanceSensor.getSignalRate();
        measurement.ambientRateKcps = distanceSensor.getAmbientRate();
        measurement.distanceMm = distanceSensor.getDistance();
        measurement.rangeStatus = distanceSensor.getRangeStatus();
        measurement.attempts = attempt + 1;
        distanceSensor.clearInterrupt();

        if (measurement.rangeStatus == 0) {
          break;
        }
      }

      const uint16_t nextZoneIndex = (zoneIndex + 1) % ZONE_COUNT;
      const uint8_t nextZoneRow = nextZoneIndex / ACTIVE_GRID_COLS;
      const uint8_t nextZoneCol = nextZoneIndex % ACTIVE_GRID_COLS;
      distanceSensor.setROI(ROI_WIDTH, ROI_HEIGHT, centerSpadForZone(nextZoneRow, nextZoneCol));

      if (!firstZone) {
        Serial.print(',');
      }
      firstZone = false;

      const uint8_t topRow = (zoneRow + EDGE_CLIP_Y) * ROI_STEP_Y;
      const uint8_t leftCol = (zoneCol + EDGE_CLIP_X) * ROI_STEP_X;
      const uint8_t centerRow = centerRowFromTop(topRow);
      const uint8_t centerCol = centerColFromLeft(leftCol);
      const uint8_t centerSpad = SPAD_MAP[centerRow][centerCol];

      Serial.print(F("{\"zone\":"));
      Serial.print(zoneIndex);
      Serial.print(F(",\"grid_row\":"));
      Serial.print(zoneRow);
      Serial.print(F(",\"grid_col\":"));
      Serial.print(zoneCol);
      Serial.print(F(",\"spad_center\":"));
      Serial.print(centerSpad);
      Serial.print(F(",\"signal_kcps\":"));
      Serial.print(measurement.signalRateKcps);
      Serial.print(F(",\"ambient_kcps\":"));
      Serial.print(measurement.ambientRateKcps);
      Serial.print(F(",\"distance_mm\":"));
      Serial.print(measurement.distanceMm);
      Serial.print(F(",\"status\":"));
      Serial.print(measurement.rangeStatus);
      Serial.print(F(",\"attempts\":"));
      Serial.print(measurement.attempts);
      if (!REPORT_INVALID_ZONES && measurement.rangeStatus != 0) {
        Serial.print(F(",\"valid\":false"));
      }
      Serial.print(F("}"));
    }
  }

  distanceSensor.stopRanging();
  Serial.print(F("],\"frame_time_ms\":"));
  Serial.print(millis() - frameStartMs);
  Serial.println(F("}"));
}

void printHelp() {
  Serial.println(F("{\"type\":\"help\",\"commands\":[\"capture\",\"meta\",\"help\"]}"));
}

void handleCommand(const char *command) {
  if (strcmp(command, "capture") == 0 || strcmp(command, "c") == 0) {
    printState(F("capturing"));
    printFrame();
    printState(F("idle"));
    return;
  }

  if (strcmp(command, "meta") == 0 || strcmp(command, "m") == 0) {
    printMeta();
    printState(F("idle"));
    return;
  }

  if (strcmp(command, "help") == 0 || strcmp(command, "?") == 0) {
    printHelp();
    printState(F("idle"));
    return;
  }

  Serial.print(F("{\"type\":\"error\",\"message\":\"Unknown command: "));
  Serial.print(command);
  Serial.println(F("\"}"));
}

void pollSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      commandBuffer[commandLength] = '\0';
      if (commandLength > 0) {
        handleCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }

    if (commandLength < (sizeof(commandBuffer) - 1)) {
      commandBuffer[commandLength++] = incoming;
    } else {
      commandLength = 0;
      Serial.println(F("{\"type\":\"error\",\"message\":\"Command too long\"}"));
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  beginWire();
  configureSensor();
  printMeta();
  printHelp();
  printState(F("idle"));
}

void loop() {
  pollSerialCommands();
  delay(2);
}
