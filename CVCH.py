import cv2
import numpy as np
import math
from typing import Tuple, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import time

class SatellitePortDetectorUI:
    def __init__(self, master):
        self.master = master
        master.title("Satellite Port Detector")
        master.geometry("1200x800")

        # Image storage
        self.original_image = None
        self.processed_image = None

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Left image display area
        self.left_image_label = tk.Label(main_frame, width=70, height=30)
        self.left_image_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        # Right image display area
        self.right_image_label = tk.Label(main_frame, width=30, height=30)
        self.right_image_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

        # Button frame
        button_frame = tk.Frame(self.master)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        # Buttons
        buttons = [
            ("Load Image", self.load_image),
            ("Detect Rotation Angle", self.detect_rotation),
            ("Find Circle in Crops", self.find_circle_in_crops),
            ("Apply Perspective Transform", self.apply_perspective_transform),
            ("Simulate Camera Movement", self.simulate_camera_movement),
        ]

        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=lambda cmd=command: self.clear_window(cmd), height=2)
            btn.pack(side=tk.LEFT, expand=True, padx=5)

    def clear_window(self, callback):
        """Clear both image display labels and call the provided callback function."""
        self.left_image_label.config(image='')
        self.left_image_label.image = None
        self.right_image_label.config(image='')
        self.right_image_label.image = None
        callback()

    def display_image(self, image, label=None, resize_dims=(700, 500)):
        """Display an OpenCV image in a Tkinter label."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail(resize_dims)
        photo = ImageTk.PhotoImage(pil_image)
        if label is None:
            label = self.left_image_label
        label.config(image=photo)
        label.image = photo

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image)
            else:
                messagebox.showerror("Error", "Could not read the image")

    def detect_rotation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=100, param2=30, minRadius=10, maxRadius=100
        )
        if circles is not None:
            circle = np.round(circles[0][0]).astype("int")
            angle = math.degrees(math.atan2(circle[1] - self.original_image.shape[0] // 2, circle[0] - self.original_image.shape[1] // 2))
            messagebox.showinfo("Rotation Angle", f"Detected rotation angle: {angle:.2f} degrees")
        else:
            messagebox.showinfo("Rotation Angle", "No circles detected")

    def find_circle_in_crops(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        crop_size = simpledialog.askinteger("Crop Size", "Enter crop size (pixels):", initialvalue=200, minvalue=50, maxvalue=500)
        if not crop_size:
            return

        h, w = self.original_image.shape[:2]
        for y in range(0, h - crop_size, crop_size // 2):
            for x in range(0, w - crop_size, crop_size // 2):
                crop = self.original_image[y:y + crop_size, x:x + crop_size]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blurred = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(blurred, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    (x_center, y_center), radius = cv2.minEnclosingCircle(approx)
                    if len(approx) > 8 and radius > 10:
                        center = (int(x_center), int(y_center))
                        radius = int(radius)
                        cv2.circle(crop, center, radius, (0, 255, 0), 2)
                        self.display_image(crop, self.right_image_label)
                        messagebox.showinfo("Circle Detection", f"Circle found at crop: ({x}, {y})")
                        return
        messagebox.showinfo("Circle Detection", "No circles found in crops")

    def apply_perspective_transform(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        angle = simpledialog.askfloat("Perspective Angle", "Enter angle (degrees):", initialvalue=22.5, minvalue=0, maxvalue=90)
        if angle is None:
            return

        h, w = self.original_image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        self.processed_image = cv2.warpAffine(self.original_image, M, (w, h))
        self.display_image(self.processed_image)

    def simulate_camera_movement(self):
        if self.original_image is None or self.processed_image is None:
            messagebox.showwarning("Warning", "Please apply perspective transform first")
            return

        current_angle = 22.5
        while current_angle > 0:
            M = cv2.getRotationMatrix2D((self.processed_image.shape[1] // 2, self.processed_image.shape[0] // 2), current_angle, 1.0)
            transformed = cv2.warpAffine(self.original_image, M, (self.processed_image.shape[1], self.processed_image.shape[0]))
            self.display_image(transformed)
            self.master.update()
            current_angle -= 1
        messagebox.showinfo("Camera Movement", "Camera movement complete")


def main():
    root = tk.Tk()
    app = SatellitePortDetectorUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
