import cv2
import numpy as np
import time

# Classe do objeto na esteira
class SimulatedObject:
    def __init__(self, shape, color, x=0, y=200, speed=5):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y
        self.speed = speed
        self.active = True

    def update(self):
        self.x += self.speed
        if self.x > 600:
            self.active = False

    def draw(self, frame):
        if self.shape == "Círculo":
            cv2.circle(frame, (self.x, self.y), 20, self.color, -1)
        elif self.shape == "Quadrado":
            cv2.rectangle(frame, (self.x - 20, self.y - 20), (self.x + 20, self.y + 20), self.color, -1)
        elif self.shape == "Retângulo":
            cv2.rectangle(frame, (self.x - 30, self.y - 15), (self.x + 30, self.y + 15), self.color, -1)
        else:
            cv2.putText(frame, "?", (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)

# Função para obter o canal Y com base na forma
def get_channel_y(shape):
    if shape == "Círculo":
        return 120
    elif shape == "Quadrado":
        return 240
    elif shape == "Retângulo":
        return 360
    else:
        return 200

# Inicialização
cap = cv2.VideoCapture(1)  # Use 0, 1 ou o índice que corresponda ao DroidCam
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

objects_on_belt = []
last_added_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara vermelha
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_detected = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                shape = "Triângulo"
            elif len(approx) == 4:
                ar = w / float(h)
                shape = "Quadrado" if 0.95 < ar < 1.05 else "Retângulo"
            elif len(approx) > 4:
                shape = "Círculo"
            else:
                shape = "Indefinido"

            shape_detected = shape

            # Marcar na imagem
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            break  # Considera só o primeiro objeto por frame

    # Adicionar novo objeto simulado à esteira
    if shape_detected and (time.time() - last_added_time > 1):
        cor = (0, 255, 255)  # Amarelo
        y_pos = get_channel_y(shape_detected)
        objects_on_belt.append(SimulatedObject(shape_detected, cor, y=y_pos))
        last_added_time = time.time()

    # Atualizar e desenhar objetos na esteira
    for obj in objects_on_belt:
        obj.update()
        obj.draw(frame)

    # Remover objetos que já passaram do final
    objects_on_belt = [obj for obj in objects_on_belt if obj.active]

    # Desenhar canais de desvio
    cv2.line(frame, (600, 0), (600, 480), (255, 255, 255), 2)
    cv2.putText(frame, "Canal 1: Circulos",   (610, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Canal 2: Quadrados",  (610, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Canal 3: Retangulos", (610, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar tela
    cv2.imshow("Smart Conveyor", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
