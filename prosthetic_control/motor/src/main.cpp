/* Make sure the motors are switched ON before reflashing the code. 
I changed zip and pinch to make the same motion. You can change it in the main loop().*/

#include <Arduino.h>
#include <Servo.h>

Servo thumb;
Servo servo2;
Servo servo3;

int pos;

int mindeg = 0;
int maxdeg = 180;
int maxthumb = 90;

int thumbPos = 0;
int servo2Pos = 0;
int servo3Pos = 0;

// 0 = neutral, 1, 3 = pinch and zip, 2 = grasp, 4 = clench unclench, 5 = move thumb
int gesture = 0;

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
  moveSmooth(mindeg, mindeg, mindeg, 5);
}

void pinch() {
  moveSmooth(maxthumb, servo2Pos, servo3Pos, 5); // move thumb first
  delay(100);

  moveSmooth(maxthumb, maxdeg, servo3Pos, 5); 

  moveSmooth(maxthumb, maxdeg, 20, 5); // keep ring/pinky semi-open
}

void grasp() {
  moveSmooth(maxthumb, maxdeg, maxdeg, 5);
}

void move_thumb() {
    for (pos = mindeg; pos <= maxthumb; pos += 1) {
    thumb.write(pos);
    delay(15);
  }
  for (pos = maxthumb; pos >= mindeg; pos -= 1) {
    thumb.write(pos);
    delay(15);
  }
}

void clench_unclench() {
  for (pos = mindeg; pos <= maxdeg; pos += 1) {
    thumb.write(pos);
    servo2.write(pos);
    servo3.write(pos);
    delay(15);
  }
  for (pos = maxdeg; pos >= mindeg; pos -= 1) {
    thumb.write(pos);
    servo2.write(pos);
    servo3.write(pos);
    delay(15);
  }
}

void setup() {
  Serial.begin(9600);
  
  thumb.attach(8);
  servo2.attach(10);
  servo3.attach(12);

  thumb.write(mindeg);
  servo2.write(mindeg);
  servo3.write(mindeg);

  // sync position with neutral
  thumbPos = mindeg;
  servo2Pos = mindeg;
  servo3Pos = mindeg;
  
  // Signal ready to Python
  Serial.println("Arduino Ready");
}

void loop() {

  // Read incoming data from Python (single byte)
  if (Serial.available() > 0) {
    int incoming = Serial.read();  // Changed from parseInt() to read()
    
    // Validate command (0-3 from Python)
    if (incoming >= 0 && incoming <= 3) {
      gesture = incoming;
      
      // Echo for debugging
      Serial.print("Received: ");
      Serial.println(gesture);
    }
  }

  if (gesture != last_gesture) {

    moveSmooth(mindeg, mindeg, mindeg, 10);
    delay(150);

    if (gesture == 0) neutral();
    else if (gesture == 1) pinch();
    else if (gesture == 2) grasp();
    else if (gesture == 3) pinch();  // Using for "Zipping" gesture
    else if (gesture == 4) clench_unclench(); // for testing
    else if (gesture == 5) move_thumb();      // for testing

    last_gesture = gesture;
  }
}