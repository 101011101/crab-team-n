/**********************************************************************
  scout_bot.ino  --  Robotic Scout Swarm worker firmware
  Board: Freenove 4WD Car for ESP32 (ESP32-WROVER CAM)

  Differences from the stock 05.3_Multi_Functional_Car sketch:

    1. Station mode, not AP mode.
       The stock sketch calls WiFi.softAP(), so each car makes its own
       network. A laptop joins one network at a time, so two cars in AP
       mode cannot both be reached. Here both cars join YOUR router.

    2. Camera enabled.
       OV2640 at QVGA/JPEG, streamed as MJPEG.

    3. Two servers on two ports.
       WebServer (port 80) is synchronous: it serves one request to
       completion before the next. An MJPEG response never completes, so
       a stream on port 80 would block every drive command and the
       watchdog below would stop the motors. The stream therefore runs on
       a separate esp_http_server at port 81.

    4. Proportional drive.
       /cmd?l=<speed>&r=<speed> instead of fixed-speed W/A/S/D, so the
       laptop can steer smoothly toward a target.

  Endpoints
    port 80   /cmd?l=&r=   set left/right track speed, -4095..4095
              /stop        stop now
              /status      JSON: id, uptime, battery, command age
              /            manual WASD page, for debugging
    port 81   /stream      MJPEG multipart stream
              /jpg         single JPEG frame (fallback if stream misbehaves)

  SET BEFORE FLASHING: WIFI_SSID, WIFI_PASS, BOT_ID, and the static IP block.
**********************************************************************/

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "esp_http_server.h"

#include "camera_pins.h"
#include "Freenove_4WD_Car_For_ESP32.h"
#include "Freenove_WS2812_Lib_for_ESP32.h"


// ============================================================
// CONFIGURE ME
// ============================================================

// Flash each car with a different BOT_ID. It sets the last octet of the
// static IP, so bot 1 is 192.168.1.51 and bot 2 is 192.168.1.52.
#define BOT_ID 1

const char* WIFI_SSID = "YOUR_ROUTER_SSID";
const char* WIFI_PASS = "YOUR_ROUTER_PASSWORD";

// Static IPs keep the laptop config stable across reboots. Change the
// first three octets to match your router (check the laptop's own IP).
// Set USE_STATIC_IP to false to fall back to DHCP; the assigned address
// is printed on the serial monitor at 115200 baud.
#define USE_STATIC_IP true

IPAddress staticIP(192, 168, 1, 50 + BOT_ID);
IPAddress gateway (192, 168, 1, 1);
IPAddress subnet  (255, 255, 255, 0);
IPAddress dns     (192, 168, 1, 1);


// ============================================================
// SAFETY WATCHDOG
// ============================================================

// The laptop resends the current drive command continuously. If commands
// stop arriving -- laptop crash, Wi-Fi drop, operator stop -- the motors
// stop on their own. This is the "stop safely on command timeout"
// requirement, and it is enforced here on the robot, not on the laptop.

const unsigned long COMMAND_TIMEOUT_MS = 500;

volatile unsigned long lastCommandTime = 0;
volatile bool driving = false;

int currentLeft  = 0;
int currentRight = 0;

// Camera pan/tilt. Servo 1 pans, servo 2 tilts. The Freenove driver
// constrains tilt to 80..180, so 90 is the lowest useful level position.
int camPan  = 90;
int camTilt = 90;


// ============================================================
// HARDWARE
// ============================================================

// Status ring: blue = searching, green = target in view, red = stopped.
Freenove_ESP32_WS2812 strip = Freenove_ESP32_WS2812(12, 32, 0, TYPE_GRB);

WebServer server(80);
httpd_handle_t stream_httpd = NULL;

unsigned long bootTime = 0;


// ============================================================
// MOTORS
// ============================================================

// From the stock sketch: a left turn drives M1/M2 backward and M3/M4
// forward, so M1+M2 are the left track and M3+M4 are the right track.

void driveTracks(int left, int right)
{
  left  = constrain(left,  -4095, 4095);
  right = constrain(right, -4095, 4095);

  currentLeft  = left;
  currentRight = right;

  Motor_Move(left, left, right, right);
}


void stopMotors()
{
  driveTracks(0, 0);
  driving = false;
}


byte statusR = 255, statusG = 255, statusB = 255;

void aimCamera(int pan, int tilt)
{
  camPan  = constrain(pan,  0, 180);
  camTilt = constrain(tilt, 80, 180);

  Servo_1_Angle(camPan);
  Servo_2_Angle(camTilt);
}


void setStatusColor(byte r, byte g, byte b)
{
  // loop() runs thousands of times a second. Refreshing the WS2812 ring on
  // every pass monopolises the RMT peripheral and starves the web server,
  // which shows up as laggy, jerky driving. Only write on a real change.
  if (r == statusR && g == statusG && b == statusB)
  {
    return;
  }

  statusR = r;
  statusG = g;
  statusB = b;

  for (int i = 0; i < 12; i++)
  {
    strip.setLedColorData(i, r, g, b);
  }

  strip.show();
}


// ============================================================
// CAMERA
// ============================================================

// QVGA (320x240) is deliberate. Green-block detection does not need more
// pixels, and two bots streaming larger frames is the fastest way to
// saturate the Wi-Fi link -- the risk called out in the PRD.

bool startCamera()
{
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0       = CAM_PIN_D0;
  config.pin_d1       = CAM_PIN_D1;
  config.pin_d2       = CAM_PIN_D2;
  config.pin_d3       = CAM_PIN_D3;
  config.pin_d4       = CAM_PIN_D4;
  config.pin_d5       = CAM_PIN_D5;
  config.pin_d6       = CAM_PIN_D6;
  config.pin_d7       = CAM_PIN_D7;

  config.pin_xclk     = CAM_PIN_XCLK;
  config.pin_pclk     = CAM_PIN_PCLK;
  config.pin_vsync    = CAM_PIN_VSYNC;
  config.pin_href     = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn     = CAM_PIN_PWDN;
  config.pin_reset    = CAM_PIN_RESET;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;            // lower number = better quality
  // Two buffers in PSRAM let capture and transmit overlap. Without PSRAM
  // that allocation fails outright, so fall back to a single DRAM buffer.
  if (psramFound())
  {
    config.fb_count    = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode   = CAMERA_GRAB_LATEST;
  }
  else
  {
    config.fb_count    = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode   = CAMERA_GRAB_WHEN_EMPTY;
    Serial.println("No PSRAM found: single buffer, expect a lower frame rate.");
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK)
  {
    Serial.printf("Camera init FAILED: 0x%x\n", err);
    Serial.println("Check camera_pins.h against your working camera sketch.");
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();

  // Fixed exposure and white balance matter more than image beauty here:
  // auto white balance shifts hues frame to frame, which makes a single
  // HSV green threshold on the laptop drift in and out of matching.
  if (s != NULL)
  {
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 0);
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
    s->set_saturation(s, 1);
  }

  Serial.println("Camera ready.");
  return true;
}


// ------------------------------------------------------------
// MJPEG STREAM HANDLER (port 81)
// ------------------------------------------------------------

static const char* STREAM_CONTENT_TYPE =
  "multipart/x-mixed-replace;boundary=frame";

static const char* STREAM_BOUNDARY =
  "\r\n--frame\r\n";

static const char* STREAM_PART =
  "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";


static esp_err_t stream_handler(httpd_req_t* req)
{
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);

  if (res != ESP_OK)
  {
    return res;
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true)
  {
    camera_fb_t* fb = esp_camera_fb_get();

    if (fb == NULL)
    {
      Serial.println("Frame grab failed");
      res = ESP_FAIL;
      break;
    }

    char part[64];

    size_t len = snprintf(part, sizeof(part), STREAM_PART, fb->len);

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));

    if (res == ESP_OK)
    {
      res = httpd_resp_send_chunk(req, part, len);
    }

    if (res == ESP_OK)
    {
      res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);

    if (res != ESP_OK)
    {
      // Client disconnected. Normal when the laptop restarts.
      break;
    }
  }

  return res;
}


// ------------------------------------------------------------
// SINGLE FRAME HANDLER (port 81)
// ------------------------------------------------------------

static esp_err_t jpg_handler(httpd_req_t* req)
{
  camera_fb_t* fb = esp_camera_fb_get();

  if (fb == NULL)
  {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);

  esp_camera_fb_return(fb);

  return res;
}


void startStreamServer()
{
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.server_port  = 81;
  config.ctrl_port    = 32769;   // must differ from the default server's
  config.max_uri_handlers = 4;

  httpd_uri_t stream_uri = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = stream_handler,
    .user_ctx = NULL
  };

  httpd_uri_t jpg_uri = {
    .uri      = "/jpg",
    .method   = HTTP_GET,
    .handler  = jpg_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK)
  {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &jpg_uri);

    Serial.println("Stream server started on port 81.");
  }
  else
  {
    Serial.println("ERROR: stream server failed to start.");
  }
}


// ============================================================
// MANUAL DEBUG PAGE (port 80)
// ============================================================

const char MANUAL_page[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scout Bot</title>
<style>
body{font-family:Arial,sans-serif;text-align:center;background:#eee;padding:20px}
button{width:100px;height:80px;margin:5px;font-size:26px;font-weight:bold;
border:none;border-radius:12px;background:#333;color:#fff}
button:active{background:#777}.stop{background:#c00}
img{margin-top:15px;border:2px solid #333;border-radius:8px}
#s{margin-top:15px;font-size:18px;font-weight:bold}
</style></head><body>
<h2>Scout Bot</h2><p>Manual override. Hold WASD or the buttons.</p>
<div>
<button onpointerdown="go(2000,2000)" onpointerup="halt()" onpointerleave="halt()">W</button>
</div><div>
<button onpointerdown="go(-1800,1800)" onpointerup="halt()" onpointerleave="halt()">A</button>
<button class="stop" onclick="halt()">STOP</button>
<button onpointerdown="go(1800,-1800)" onpointerup="halt()" onpointerleave="halt()">D</button>
</div><div>
<button onpointerdown="go(-2000,-2000)" onpointerup="halt()" onpointerleave="halt()">S</button>
</div>
<div id="s">STOPPED</div>
<img id="cam" width="320" height="240">
<script>
let cur=null,ver=0;
document.getElementById("cam").src="http://"+location.hostname+":81/stream";
async function send(l,r){
  try{await fetch("/cmd?l="+l+"&r="+r,{cache:"no-store"});
  document.getElementById("s").innerText="L "+l+"  R "+r;}catch(e){console.log(e)}
}
function go(l,r){
  const k=l+","+r; if(cur===k)return; cur=k; ver++;
  const v=ver; send(l,r); beat(k,v);
}
async function beat(k,v){
  // The robot stops itself after 500 ms of silence, so hold = resend.
  while(cur===k&&ver===v){
    await new Promise(r=>setTimeout(r,200));
    if(cur!==k||ver!==v)return;
    await send(...k.split(","));
  }
}
function halt(){
  cur=null; ver++;
  fetch("/stop",{cache:"no-store",keepalive:true});
  document.getElementById("s").innerText="STOPPED";
}
document.addEventListener("keydown",e=>{
  if(e.repeat)return; const k=e.key.toLowerCase();
  if(k==="w")go(2000,2000); else if(k==="s")go(-2000,-2000);
  else if(k==="a")go(-1800,1800); else if(k==="d")go(1800,-1800);
  else if(e.code==="Space"){e.preventDefault();halt();}
});
document.addEventListener("keyup",e=>{
  if("wasd".includes(e.key.toLowerCase()))halt();
});
window.addEventListener("blur",halt);
window.addEventListener("beforeunload",halt);
</script></body></html>
)rawliteral";


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("==================================");
  Serial.printf ("Scout Bot %d starting\n", BOT_ID);
  Serial.println("==================================");

  // ----- hardware -----

  PCA9685_Setup();
  Buzzer_Setup();

  strip.begin();
  strip.setBrightness(10);
  setStatusColor(255, 0, 0);          // red until we are online

  stopMotors();
  aimCamera(camPan, camTilt);

  // ----- camera -----
  // Started before Wi-Fi: camera init is the failure most likely to need
  // a pin fix, and it is easier to see that on a quiet serial log.

  if (!startCamera())
  {
    Serial.println("Continuing without camera. Drive commands still work.");
  }

  // ----- wifi (station mode) -----

  WiFi.mode(WIFI_STA);

  if (USE_STATIC_IP)
  {
    if (!WiFi.config(staticIP, gateway, subnet, dns))
    {
      Serial.println("Static IP config rejected; falling back to DHCP.");
    }
  }

  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.printf("Joining %s", WIFI_SSID);

  unsigned long wifiStart = millis();

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(400);
    Serial.print(".");

    // A car stuck in a connect loop looks identical to a dead car. Reboot
    // so it retries cleanly instead of sitting silent.
    if (millis() - wifiStart > 20000)
    {
      Serial.println();
      Serial.println("Could not join Wi-Fi in 20 s. Restarting.");
      ESP.restart();
    }
  }

  Serial.println();
  Serial.println("Wi-Fi connected.");
  Serial.print  ("BOT IP: ");
  Serial.println(WiFi.localIP());
  Serial.printf ("Commands: http://%s/cmd?l=0&r=0\n", WiFi.localIP().toString().c_str());
  Serial.printf ("Stream:   http://%s:81/stream\n",   WiFi.localIP().toString().c_str());

  // Quieter Wi-Fi power saving costs a little current but removes the
  // latency spikes that make a reactive controller oscillate.
  WiFi.setSleep(false);

  // ----- command server (port 80) -----

  server.on("/", []()
  {
    server.send_P(200, "text/html", MANUAL_page);
  });

  server.on("/cmd", []()
  {
    int l = server.hasArg("l") ? server.arg("l").toInt() : 0;
    int r = server.hasArg("r") ? server.arg("r").toInt() : 0;

    driveTracks(l, r);

    driving         = (l != 0 || r != 0);
    lastCommandTime = millis();

    server.send(200, "text/plain", "ok");
  });

  server.on("/stop", []()
  {
    stopMotors();
    server.send(200, "text/plain", "stopped");
  });

  // Camera aim. Absolute angles, not deltas: the laptop holds the
  // desired pose, so a dropped request cannot accumulate an offset.
  server.on("/cam", []()
  {
    int pan  = server.hasArg("pan")  ? server.arg("pan").toInt()  : camPan;
    int tilt = server.hasArg("tilt") ? server.arg("tilt").toInt() : camTilt;

    aimCamera(pan, tilt);

    char body[48];
    snprintf(body, sizeof(body), "{\"pan\":%d,\"tilt\":%d}", camPan, camTilt);

    server.send(200, "application/json", body);
  });


  server.on("/status", []()
  {
    char body[192];

    snprintf(body, sizeof(body),
      "{\"id\":%d,\"uptime_ms\":%lu,\"battery_v\":%.2f,"
      "\"cmd_age_ms\":%lu,\"left\":%d,\"right\":%d,\"driving\":%s,"
      "\"pan\":%d,\"tilt\":%d}",
      BOT_ID,
      millis() - bootTime,
      Get_Battery_Voltage(),
      millis() - lastCommandTime,
      currentLeft,
      currentRight,
      driving ? "true" : "false",
      camPan,
      camTilt);

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", body);
  });

  server.begin();
  Serial.println("Command server started on port 80.");

  // ----- stream server (port 81) -----

  startStreamServer();

  setStatusColor(0, 0, 255);          // blue: online, idle
  Buzzer_Alert(1, 1);

  bootTime        = millis();
  lastCommandTime = millis();
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  server.handleClient();

  // Watchdog. Runs regardless of what the laptop believes.
  if (driving && (millis() - lastCommandTime > COMMAND_TIMEOUT_MS))
  {
    stopMotors();
    setStatusColor(255, 0, 0);
    Serial.println("Command timeout -> motors stopped");
  }
  else if (driving)
  {
    setStatusColor(0, 255, 0);        // green: under laptop control
  }
}
