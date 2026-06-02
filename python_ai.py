from deepface import DeepFace
import cv2
import serial
import time
from collections import deque


ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)


BUFFER_SIZE = 5
CONF_THRESHOLD = 40
SEND_DELAY = 0.4
EMOTION_INTERVAL = 0.7
NO_FACE_TIMEOUT = 10

buffer = deque(maxlen=BUFFER_SIZE)
last_sent = ""
last_send_time = 0
last_emotion_time = 0
last_face_time = time.time()
manual_off = False


emotion_colors = {
    "happy": (255, 180, 0),
    "sad": (0, 0, 255),
    "angry": (255, 20, 0),
    "neutral": (255, 255, 255)
}

def get_stable_emotion(new_emotion):
    buffer.append(new_emotion)
    return max(set(buffer), key=buffer.count)

# 🔌 Send to ESP32
def send_to_esp32(emotion):
    global last_sent, last_send_time

    if manual_off:
        return

    now = time.time()

    if emotion != last_sent and (now - last_send_time) > SEND_DELAY:
        r, g, b = emotion_colors[emotion]
        data = f"{r},{g},{b}\n"

        try:
            ser.write(data.encode())
            print(f"✅ {emotion} → {data.strip()}")
            last_sent = emotion
            last_send_time = now
        except:
            print("⚠️ Serial error")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

try:
    stable_emotion = "neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (360, 270))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

       
        for (x, y, w, h) in faces:
            scale_x = frame.shape[1] / 360
            scale_y = frame.shape[0] / 270

            X = int(x * scale_x)
            Y = int(y * scale_y)
            W = int(w * scale_x)
            H = int(h * scale_y)

            cv2.rectangle(frame, (X, Y), (X+W, Y+H), (0, 255, 0), 2)

        
        if len(faces) > 0:
            last_face_time = time.time()
        else:
            if time.time() - last_face_time > NO_FACE_TIMEOUT:
                ser.write(("off\n").encode())
                stable_emotion = "neutral"
                cv2.putText(frame, "No face → LEDs OFF", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

      
        if (time.time() - last_emotion_time > EMOTION_INTERVAL) and len(faces) > 0:
            try:
                (x, y, w, h) = faces[0]
                face_roi = small_frame[y:y+h, x:x+w]

                result = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )

                emotions = result[0]['emotion']

                filtered = {
                    k: emotions[k]
                    for k in ["happy", "sad", "angry", "neutral"]
                }

                
                happy_score = filtered["happy"]
                angry_score = filtered["angry"]

               
                filtered["happy"] *= 1.5

                filtered["angry"] *= 1.4

               
                if angry_score > 25:
                    detected = "angry"
                    confidence = angry_score

                elif happy_score > 15:
                    detected = "happy"
                    confidence = happy_score

               
                else:
                    detected = max(filtered, key=filtered.get)
                    confidence = filtered[detected]

                if confidence > CONF_THRESHOLD:
                    stable_emotion = get_stable_emotion(detected)
                    print(f"{detected} ({confidence:.1f}%) → {stable_emotion}")

                last_emotion_time = time.time()

            except Exception as e:
                print(" DeepFace error:", e)

        # 🔌 Send output
        send_to_esp32(stable_emotion)

        # 🖥️ UI
        cv2.putText(frame, f"Emotion: {stable_emotion}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        status = "OFF (Manual)" if manual_off else f"Active: {last_sent}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Emotion-Based Mood Lighting", frame)

        key = cv2.waitKey(10)

        if key != -1:
            key = key & 0xFF

            if key == 27:  # ESC
                break

            elif key == ord('o'):  # OFF
                manual_off = True
                ser.write(("off\n").encode())
                print("🔴 Manual OFF")

            elif key == ord('r'):  # Resume
                manual_off = False
                last_sent = ""
                print("🟢 Resumed")

            elif key == ord('k'):  # Kill
                print(" Program terminated")
                break

except KeyboardInterrupt:
    print("\n Stopping...")

finally:
    print(" Cleaning up...")

    try:
        ser.write(("off\n").encode())
        time.sleep(0.5)
    except:
        pass

    ser.close()
    cap.release()
    cv2.destroyAllWindows()
