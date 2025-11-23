#include <Servo.h>
#include <LiquidCrystal_I2C.h>
// SDA (A4) || SCL (A5)
// La dirección I2C (0x27) y las dimensiones (16 columnas, 2 filas)
LiquidCrystal_I2C lcd(0x27, 16, 2); 
bool newData = false;
Servo miServo;
const int pinServo = 9; 
const int TOUCH_SENSOR_PIN = 2;


void setup() {
  miServo.attach(pinServo);
  Serial.begin(9600);  
  pinMode(TOUCH_SENSOR_PIN, INPUT);
  
  miServo.write(0); 

  // Inicializar el LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();

  Serial.println("Arduino listo. Envia texto por el Monitor Serial.");
}

void loop() {

  // Mensaje inicial de bienvenida
  lcd.setCursor(0, 0);
  lcd.print("Bienvenido!");
  lcd.setCursor(0, 1);
  lcd.print("Hotel AUTMTZDO");
  
  // Comprueba si hay datos disponibles en el puerto serial
  if (Serial.available()) {
    // Lee toda la línea de entrada hasta el carácter de nueva línea
    String command = Serial.readStringUntil('\n');
    
    // Convertimos a minúsculas y eliminamos espacios por si acaso
    command.toLowerCase();
    command.trim();
    
    if (command == "door:true") {
      miServo.write(180);
      Serial.println("Abriendo puerta");
      delay(5000);
      miServo.write(0);
    }

    if (command.startsWith("display")){
      Serial.println(command);
      // 1. Siempre se intenta leer el dato serial
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
        
        // 3. Imprime la cadena en el monitor serial para confirmación
        Serial.print("Mostrando en LCD: ");
        Serial.println(command);
        
        // 4. Se reinicia la bandera DESPUÉS de mostrar el mensaje
        newData = false; 
        delay(5000);
        lcd.clear(); 
      }

    }

    int touchState = digitalRead(TOUCH_SENSOR_PIN);
    if (touchState == HIGH) Serial.println("camera:true");

    }
