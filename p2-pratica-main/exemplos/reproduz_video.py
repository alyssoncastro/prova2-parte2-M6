import cv2 #importanto

# Carrega o modelo pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Abre o arquivo de vídeo
video_capture = cv2.VideoCapture('../assets/arsene.mp4')



# Loop de leitura frame por frame
while True:
    # Lê um frame do vídeo e guarda o resultado da leitura
    ret, frame = video_capture.read()
    # Se não conseguiu ler o frame, para o loop
    if not ret:
        break

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Realiza a detecção de faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Marca os retângulos correspondentes a cada face encontrada
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Exibe o frame
    cv2.imshow('Video Playback', frame)
    # Se o usuário apertar 'q', encerra o playback
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break



# Libera os recursos
video_capture.release()
cv2.destroyAllWindows()
