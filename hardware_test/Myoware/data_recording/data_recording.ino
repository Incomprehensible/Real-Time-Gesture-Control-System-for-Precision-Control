#include <MyoWare.h>

MyoWare myoware;

void setup() 
{
  Serial.begin(115200);
  while(!Serial);

  myoware.setENVPin(A0);                      // Arduino pin connected to ENV
  //myoware.setRAWPin(A0);
  pinMode(myoware.getStatusLEDPin(), OUTPUT); // initialize the built-in LED pin to indicate 
                                              // when a central is connected
}

void loop() 
{
  int testValue = myoware.readSensorOutput(MyoWare::ENVELOPE);
  Serial.println(testValue);
}
