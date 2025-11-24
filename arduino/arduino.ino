#include <Servo.h>
#include <LiquidCrystal_I2C.h>
// SDA (A4) || SCL (A5)
// La dirección I2C (0x27) y las dimensiones (16 columnas, 2 filas)
LiquidCrystal_I2C lcd(0x27, 16, 2); 
bool newData = false;
Servo miServo;
const int pinServo = 9; 
const int TOUCH_SENSOR_PIN = 2;

// Variables para control de tiempo sin bloqueo (non-blocking)
unsigned long servoStartTime = 0;
unsigned long lcdStartTime = 0;
bool servoActive = false;
bool lcdActive = false;
const unsigned long SERVO_DURATION = 5000; // 5 segundos
const unsigned long LCD_DURATION = 5000;   // 5 segundos

void setup() {
  miServo.attach(pinServo);
  Serial.begin(9600);  
  pinMode(TOUCH_SENSOR_PIN, INPUT);
  
  miServo.write(180); 

  // Inicializar el LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  
  // Mostrar mensaje inicial de bienvenida
  lcd.setCursor(0, 0);
  lcd.print("Bienvenido!");
  lcd.setCursor(0, 1);
  lcd.print("Hotel AUTMTZDO");

  Serial.println("Arduino listo. Envia texto por el Monitor Serial.");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Control del servo (sin bloqueo)
  if (servoActive) {
    if (currentTime - servoStartTime >= SERVO_DURATION) {
      // Tiempo completado, cerrar puerta
      miServo.write(180);
      servoActive = false;
      Serial.println("Puerta cerrada");
    }
  }
  
  // Control del LCD (sin bloqueo)
  if (lcdActive) {
    if (currentTime - lcdStartTime >= LCD_DURATION) {
      // Tiempo completado, limpiar LCD y mostrar bienvenida
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Bienvenido!");
      lcd.setCursor(0, 1);
      lcd.print("Hotel AUTMTZDO");
      lcdActive = false;
    }
  }
  
  // Comprueba si hay datos disponibles en el puerto serial
  if (Serial.available()) {
    // Lee toda la línea de entrada hasta el carácter de nueva línea
    String command = Serial.readStringUntil('\n');
    
    // Convertimos a minúsculas y eliminamos espacios por si acaso
    command.toLowerCase();
    command.trim();
    
    // Comando para abrir puerta (funciona en paralelo con LCD)
    if (command == "door:true") {
      miServo.write(90);
      servoStartTime = currentTime;
      servoActive = true;
      Serial.println("Abriendo puerta");
    }

    // Comando para mostrar en LCD (funciona en paralelo con servo)
    if (command.startsWith("display")){
      Serial.println(command);
      
      // Limpia la pantalla para eliminar cualquier texto anterior
      lcd.clear(); 
      
      // Calcula la longitud de la cadena
      int stringLength = command.length() - 8;
      
      // Muestra la primera línea (primeros 16 caracteres)
      String line1 = command.substring(8, min(stringLength+8, 24));
      lcd.setCursor(0, 0);
      lcd.print(line1);

      // Si la cadena es más larga, muestra la segunda línea (caracteres 16 a 32)
      if (stringLength > 16) {
        String line2 = command.substring(24, min(stringLength+8, 40));
        lcd.setCursor(0, 1);
        lcd.print(line2);
      }
      
      // Activar temporizador del LCD
      lcdStartTime = currentTime;
      lcdActive = true;
      
      // Imprime la cadena en el monitor serial para confirmación
      Serial.print("Mostrando en LCD: ");
      Serial.println(command);
      
      // Se reinicia la bandera DESPUÉS de mostrar el mensaje
      newData = false; 
    }
  }

  // Leer sensor táctil
  int touchState = digitalRead(TOUCH_SENSOR_PIN);
  if (touchState == HIGH) {
    Serial.println("camera:true");
  }
}
