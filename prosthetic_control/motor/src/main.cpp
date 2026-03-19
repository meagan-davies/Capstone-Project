#include <Arduino.h>
#include <Servo.h>

Servo thumb;
Servo servo2;
Servo servo3;

int pos;

int mindeg = 0;
int maxdeg = 180;

int thumbPos = 0;
int servo2Pos = 0;
int servo3Pos = 0;

// 0 = neutral, 1 = pinch, 2 = grasp, 3 = clench unclench, 4 = test thumb
int gesture = 3;

int last_gesture = -1;

void moveSmooth(int targetThumb, int targetS2, int targetS3, int stepDelay) {

  while (thumbPos != targetThumb || servo2Pos != targetS2 || servo3Pos != targetS3) {

    if (thumbPos < targetThumb) thumbPos++;
    else if (thumbPos > targetThumb) thumbPos--;

    if (servo2Pos < targetS2) servo2Pos++;
    else if (servo2Pos > targetS2) servo2Pos--;

    if (servo3Pos < targetS3) servo3Pos++;
    else if (servo3Pos > targetS3) servo3Pos--;

    thumb.write(thumbPos);
    servo2.write(servo2Pos);
    servo3.write(servo3Pos);

    delay(stepDelay);
  }
}

void neutral() {
  moveSmooth(mindeg, mindeg, mindeg, 10);
}

void pinch() {
  moveSmooth(maxdeg, maxdeg, 90, 10);
}

void grasp() {
  moveSmooth(maxdeg, maxdeg, maxdeg, 10);
}

void move_thumb() {
    for (pos = mindeg; pos <= maxdeg; pos += 1) { // goes from 0 degrees to 180 degrees in steps of 1 degree
    thumb.write(pos);
    delay(15);                          // wait 15 ms for servo to reach position
  }
  for (pos = maxdeg; pos >= mindeg; pos -= 1) { // goes from 180 degrees to 0 degrees
    thumb.write(pos);
    delay(15);
  }
}

void clench_unclench() {
  for (pos = mindeg; pos <= maxdeg; pos += 1) { // goes from 0 degrees to 180 degrees in steps of 1 degree
    thumb.write(pos);
    servo2.write(pos);
    servo3.write(pos);
    delay(15);                          // wait 15 ms for servo to reach position
  }
  for (pos = maxdeg; pos >= mindeg; pos -= 1) { // goes from 180 degrees to 0 degrees
    thumb.write(pos);
    servo2.write(pos);
    servo3.write(pos);
    delay(15);
  }
}

void setup() {
  thumb.attach(8);
  servo2.attach(10);
  servo3.attach(12);

  moveSmooth(mindeg, mindeg, mindeg, 20);

  // Sync software with reality
  thumbPos = mindeg;
  servo2Pos = mindeg;
  servo3Pos = mindeg;
}

void loop() {
  if (gesture != last_gesture) {

    moveSmooth(mindeg, mindeg, mindeg, 10);
    delay(150);

    if (gesture == 0) neutral();
    else if (gesture == 1) pinch();
    else if (gesture == 2) grasp();
    else if (gesture == 3) clench_unclench();
    else if (gesture == 4) move_thumb();

    last_gesture = gesture;
  }
}