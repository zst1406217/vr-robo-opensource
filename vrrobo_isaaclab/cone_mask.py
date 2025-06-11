import cv2
import numpy as np

#!/usr/bin/env python3

def main():
    # Load the image
    image_path = "./obs_test.png"
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image:", image_path)
        return

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --------------------------
    # Red cone mask
    # Red is split in the HSV space near the edges, so we use two ranges.
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # --------------------------
    # Blue cone mask
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --------------------------
    # Green cone mask
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: refine the masks using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)

    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)

    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)

    # Save the resulting masks
    cv2.imwrite("red_cone_mask.png", mask_red)
    cv2.imwrite("blue_cone_mask.png", mask_blue)
    cv2.imwrite("green_cone_mask.png", mask_green)
    print("Masks saved as red_cone_mask.png, blue_cone_mask.png, and green_cone_mask.png")

if __name__ == "__main__":
    main()