/*
Obtaining muscle data from uMyo via BLE on various Arduino boards
Usage: install ArduinoBLE library, run the code on BLE enabled Arduino (nRF52 or ESP32 core)
 - open Serial Plotter (or Serial Monitor) at 115200 speed
 - turn on uMyo device
 - switch it into BLE mode if necessary (to switch, press button once or twice, depending on current mode)
 - you should see muscle activity chart on a plot (or as serial output)
*/

#include <uMyo_BLE.h>

float bins[2][4] = {{}}; 

void setup() {
  Serial.begin(115200);
  uMyo.begin();
}

void loop() 
{
  uMyo.run();
  int dev_count = uMyo.getDeviceCount(); //if more than one device is present, show all of them
  for(int d = 0; d < dev_count; d++)
  {
      uMyo.getSpectrum(d, bins[d]);
      Serial.print("Frequency spectrum for dev ");
      Serial.println(d);

    //Serial.print(uMyo.getMuscleLevel(d)); 
    for (int i=0; i<4; ++i){
      Serial.print("Bin[");
      Serial.print(i);
      Serial.print("]: ");
      Serial.println(bins[d][i]);
    }
    //if(d < dev_count-1) Serial.print(' ');
    //else Serial.println();
  }
  delay(30);
}