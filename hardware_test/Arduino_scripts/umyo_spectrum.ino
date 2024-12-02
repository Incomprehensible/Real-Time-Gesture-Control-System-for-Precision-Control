/*
Obtaining muscle data from uMyo via BLE on various Arduino boards
Usage: install ArduinoBLE library, run the code on BLE enabled Arduino (nRF52 or ESP32 core)
 - open Serial Plotter (or Serial Monitor) at 115200 speed
 - turn on uMyo device
 - switch it into BLE mode if necessary (to switch, press button once or twice, depending on current mode)
 - you should see muscle activity chart on a plot (or as serial output)
*/

#include <uMyo_BLE.h>


void setup() {
  Serial.begin(115200);
  uMyo.begin();
}

float bins[2][4] = {{}};

void loop() 
{
  uMyo.run();
  int dev_count = uMyo.getDeviceCount(); //if more than one device is present, show all of them
  int d = 0;
  //for(int d = 0; d < dev_count; d++)
  //{
    uMyo.getSpectrum(d, bins[d]);
    for (int b=1; b<4; ++b) {
     // Serial.print("Bin[");
      //Serial.print(b);
      //Serial.print("]: ");
      Serial.print(bins[d][b]);
      if (b < 3)
        Serial.print(' ');
      else
        Serial.println();
    }
    //Serial.print(uMyo.getMuscleLevel(d));
    //if(d < dev_count-1) Serial.print(' ');
    //else Serial.println();
  //}
  delay(30);
}