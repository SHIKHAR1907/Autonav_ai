import cv2
import time


class CameraInput:

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None

    def start(self):
        # cv2.CAP_DSHOW avoids MSMF hangs on Windows
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # Fallback to default backend if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Camera with ID {self.camera_id} could not be opened. "
                    "Try camera_id=1 or check camera privacy permissions."
                )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"Camera {self.camera_id} initialized successfully")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not initialized")

        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = CameraInput(camera_id=0)

    try:
        camera.start()
        prev_time = time.time()

        while True:
            frame = camera.read()

            if frame is None:
                print("Frame unavailable")
                break

            # Accurate frame-to-frame FPS
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv2.putText(
                frame,
                f"UGV Camera | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("SIH26126 - Camera Input", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(error)

    finally:
        camera.release()