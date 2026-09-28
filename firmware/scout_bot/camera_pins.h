// Camera pin map for the Freenove ESP32-WROVER CAM board (the 4WD car controller).
// VERIFY these against the Freenove camera example sketch that you already got
// a picture out of. If that sketch used a different map, copy its values here.
#ifndef _SCOUT_CAMERA_PINS_H
#define _SCOUT_CAMERA_PINS_H

#define CAM_PIN_PWDN    -1   // not wired on this board
#define CAM_PIN_RESET   -1   // not wired on this board
#define CAM_PIN_XCLK    21
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27

#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      19
#define CAM_PIN_D2      18
#define CAM_PIN_D1       5
#define CAM_PIN_D0       4

#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

#endif
