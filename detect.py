import cv2
import numpy as np
import argparse
from models import SCRFD
from utils.general import draw_corners, smooth_face


def parse_args():
    parser = argparse.ArgumentParser(description="Face smoothing script for image, video, or webcam")
    parser.add_argument("--input", type=str, default="0", help="Path to the input video/image file or '0' for webcam")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output video or image file")
    return parser.parse_args()


def process_video(input_source, output_path, is_webcam=False):
    face_detector = SCRFD(model_path="weights/det_2.5g.onnx")

    if is_webcam:
        cap = cv2.VideoCapture(0)  # Open the default webcam
    else:
        cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        raise IOError("Cannot open video or webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_webcam else 30  # Use 30 FPS for webcam

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip frame horizontally for webcam view

        if not success:
            print("Failed to obtain frame or EOF")
            break

        original_frame = frame.copy()

        boxes_list, points_list = face_detector.detect(frame)

        for boxes, points in zip(boxes_list, points_list):
            x1, y1, x2, y2, score = boxes.astype(np.int32)
            draw_corners(frame, boxes)
            face = frame[y1:y2, x1:x2]
            smoothed_face = smooth_face(face)
            frame[y1:y2, x1:x2] = smoothed_face

        concat_frame = cv2.hconcat([original_frame, frame])

        cv2.imshow("Video/Camera Demo", concat_frame)
        out.write(concat_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        raise IOError("Cannot open image")

    face_detector = SCRFD(model_path="weights/det_2.5g.onnx")
    original_image = image.copy()

    boxes_list, points_list = face_detector.detect(image)

    for boxes, points in zip(boxes_list, points_list):
        x1, y1, x2, y2, score = boxes.astype(np.int32)
        draw_corners(image, boxes)
        face = image[y1:y2, x1:x2]
        smoothed_face = smooth_face(face)
        image[y1:y2, x1:x2] = smoothed_face

    concat_image = cv2.hconcat([original_image, image])

    cv2.imwrite(output_path, concat_image)
    cv2.imshow("Image Result", concat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Check if the input is a webcam (if it's '0' or a digit representing the webcam ID)
    if args.input.isdigit():
        process_video(int(args.input), args.output, is_webcam=True)
    # Check if the input is an image (based on file extension)
    elif args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        process_image(args.input, args.output[:-4] + ".jpg")
    # Otherwise, assume the input is a video file
    else:
        process_video(args.input, args.output[:-4] + ".mp4")


if __name__ == "__main__":
    main()
