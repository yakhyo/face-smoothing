import cv2
import numpy as np

from models import SCRFD


def smooth_face(face: np.ndarray) -> np.ndarray:
    hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hsv_mask = cv2.inRange(hsv_img, np.array([0., 80., 80.]), np.array([200., 255., 255.]))
    full_mask = cv2.merge((hsv_mask, hsv_mask, hsv_mask))

    blurred_img = cv2.bilateralFilter(face, 15, 50, 50)

    masked_img = cv2.bitwise_and(blurred_img, full_mask)

    # Invert mask
    inverted_mask = cv2.bitwise_not(full_mask)

    # Anti-mask
    masked_img2 = cv2.bitwise_and(face, inverted_mask)

    # Add the masked images together
    smoothed_face = cv2.add(masked_img2, masked_img)

    return smoothed_face


def main():
    face_detector = SCRFD(model_path="weights/det_2.5g.onnx")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to obtain frame or EOF")
            break

        original_frame = frame.copy()

        boxes_list, points_list = face_detector.detect(frame, max_num=3)  # maximum three faces

        for boxes, points in zip(boxes_list, points_list):
            x1, y1, x2, y2, score = boxes.astype(np.int32)
            
            face = frame[y1:y2, x1:x2]  # crop face
            smoothed_face = smooth_face(roi)  # smooth face
            frame[y1:y2, x1:x2] = smoothed_face  # replace original with smoothed face

        concat_frame = cv2.hconcat([original_frame, frame])

        cv2.imshow("Demo", concat_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
