import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
import random
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import time

class SatellitePortDetectorUI:
    def __init__(self, master):
        self.master = master
        master.title("Satellite Port Detector")
        master.geometry("1200x800")

        # Constants for the satellite port
        self.OUTER_SQUARE_SIZE = 40  # cm
        self.INNER_SQUARE_SIZE = 30  # cm
        self.SQUARE_SPACING = 2.5    # cm
        self.CAMERA_DISTANCE = 100   # cm
        self.DEBUG = False

        # Image storage
        self.original_image = None
        self.processed_image = None

        # Create UI elements
        self.create_widgets()

    def clear_window(self):
        """Clear both image display labels and reset image states"""
        self.left_image_label.config(image='')
        self.left_image_label.image = None
        self.right_image_label.config(image='')
        self.right_image_label.image = None
        
        # Reset image tracking variables
        self.original_image = None
        self.processed_image = None

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for better feature detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        return gray, edges

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Left image display area (larger)
        self.left_image_label = tk.Label(main_frame, width=70, height=30)
        self.left_image_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        # Right image display area
        self.right_image_label = tk.Label(main_frame, width=30, height=30)
        self.right_image_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

        # Button frame with larger buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        # Buttons for different tasks
        buttons = [
            ("Clear Window", self.clear_window),
            ("Load Image", self.load_image),
            ("Detect Rotation", self.detect_rotation_angle),
            ("Find Circle", self.find_circle_in_crops),
            ("Perspective Transform", self.apply_perspective_transform),
            ("Camera Movement", self.simulate_camera_movement)
        ]

        for (text, command) in buttons:
            btn = tk.Button(button_frame, text=text, command=command, height=2)
            btn.pack(side=tk.LEFT, expand=True, padx=5)

    def display_image(self, image, label=None, resize_dims=(700, 500)):
        """
        Display an OpenCV image in a Tkinter label
        :param image: OpenCV image
        :param label: Tkinter label to display on (default to left label)
        :param resize_dims: Tuple of max width and height
        """
        # Convert OpenCV image to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Resize image to fit UI while maintaining aspect ratio
        pil_image.thumbnail(resize_dims)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        if label is None:
            label = self.left_image_label
        
        label.config(image=photo)
        label.image = photo  # Keep a reference

    def load_image(self):
        # Clear previous displays
        self.clear_window()

        # Open file dialog to choose image
        file_path = filedialog.askopenfilename(
            title="Select Image", 
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            # Read image with OpenCV
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is not None:
                # Display original image
                self.display_image(self.original_image)
            else:
                messagebox.showerror("Error", "Could not read the image")

    def detect_rotation_angle(self):
        # Validate image
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        try:
            # Detect rotation angle
            angle = self._detect_rotation()
            
            if angle is not None:
                # Create rotated image
                center = (self.original_image.shape[1] // 2, self.original_image.shape[0] // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(
                    self.original_image, 
                    rot_matrix, 
                    (self.original_image.shape[1], self.original_image.shape[0])
                )
                
                # Display original and rotated images
                self.display_image(self.original_image)
                self.display_image(rotated_image, self.right_image_label)
                
                # Add text overlay with angle
                display_img = self.original_image.copy()
                cv2.putText(display_img, 
                            f"Rotation Angle: {angle:.2f} degrees", 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 255, 0), 
                            2
                )
                self.display_image(display_img)
                
                messagebox.showinfo("Rotation", f"Detected rotation angle: {angle:.2f} degrees")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _detect_rotation(self) -> Optional[float]:
        """Detect rotation angle using circle marker and squares."""
        gray, _ = self._preprocess_image(self.original_image)
        
        # Detect circles using adaptive parameters
        circles = None
        dp_values = [1.2, 1.5, 1.8]
        param2_values = [30, 20, 15]
        
        for dp in dp_values:
            for param2 in param2_values:
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=dp, 
                    minDist=50, param1=100, param2=param2,
                    minRadius=10, maxRadius=int(min(self.original_image.shape[:2])/4)
                )
                if circles is not None:
                    break
            if circles is not None:
                break
                
        if circles is None:
            messagebox.showwarning("Warning", "No circle detected")
            return None
            
        # Get the circle closest to any corner
        circles = np.round(circles[0, :]).astype("int")
        h, w = self.original_image.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        
        min_dist = float('inf')
        circle_idx = 0
        
        for idx, (cx, cy, _) in enumerate(circles):
            for corner in corners:
                dist = np.sqrt((cx - corner[0])**2 + (cy - corner[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    circle_idx = idx
        
        cx, cy, r = circles[circle_idx]
        angle = math.degrees(math.atan2(cy, cx))
        
        if self.DEBUG:
            debug_img = self.original_image.copy()
            cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)
            cv2.imshow("Circle Detection", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return angle

    def find_circle_in_crops(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        try:
            # Get crop size from user
            crop_size = simpledialog.askinteger(
                "Crop Size", 
                "Enter crop size (width/height in pixels):", 
                initialvalue=200, 
                minvalue=50, 
                maxvalue=500
            )
            
            if crop_size:
                # Animate crop search
                self.animate_crop_search(self.original_image, crop_size)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def animate_crop_search(self, image, crop_size):
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size, crop_size
        
        # Calculate optimal step size
        step_size = min(crop_h // 2, crop_w // 2)
        
        # Animate search
        for y in range(0, h - crop_h, step_size):
            for x in range(0, w - crop_w, step_size):
                # Create a copy of the image to draw rectangle
                search_image = image.copy()
                cv2.rectangle(search_image, (x, y), (x + crop_w, y + crop_h), (0, 255, 0), 2)
                
                # Display search area
                self.display_image(search_image)
                
                # Extract current crop
                crop = image[y:y + crop_h, x:x + crop_w]
                
                # Display current crop
                self.display_image(crop, self.right_image_label)
                
                # Process current crop
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 1)
                
                # Detect circles
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=30, param1=100, param2=20,
                    minRadius=10, maxRadius=min(crop_h, crop_w)//2
                )
                
                # Update UI
                self.master.update()
                time.sleep(0.1)  # Slow down animation
                
                # If circle found
                if circles is not None:
                    # Get the first detected circle
                    circle = np.round(circles[0][0]).astype("int")
                    circle_x, circle_y, radius = circle

                    # Draw circle on crop
                    crop_with_circle = crop.copy()
                    cv2.circle(crop_with_circle, (circle_x, circle_y), radius, (0, 255, 0), 2)
                    
                    # Display final results
                    self.display_image(search_image)
                    self.display_image(crop_with_circle, self.right_image_label)
                    
                    messagebox.showinfo(
                        "Circle Detection", 
                        f"Circle found at crop position: (x={x}, y={y})\n"
                        f"Circle center in crop: (x={circle_x}, y={circle_y})"
                    )
                    return (x, y)
        
        messagebox.showinfo("Circle Detection", "No circle found in any crop")
        return None

    def apply_perspective_transform(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        try:
            # Get angle from user
            angle = simpledialog.askfloat(
                "Perspective Angle", 
                "Enter viewing angle (in degrees):", 
                initialvalue=22.5, 
                minvalue=0, 
                maxvalue=90
            )
            
            if angle is not None:
                # Apply perspective transform
                transformed, _ = self._apply_perspective_transform(
                    self.original_image, angle
                )
                
                # Display transformed image
                self.processed_image = transformed
                self.display_image(transformed)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _apply_perspective_transform(self, image: np.ndarray, angle: float = 22.5):
        """
        Apply perspective transform to simulate view from right side.
        angle: viewing angle from right side in degrees
        """
        h, w = image.shape[:2]
        
        # Calculate camera position relative to port center
        Z = self.CAMERA_DISTANCE * math.cos(math.radians(angle))  # depth
        X = self.CAMERA_DISTANCE * math.sin(math.radians(angle))  # right side offset
        
        # Calculate focal length based on image size and FOV
        fov = 60  # assumed field of view in degrees
        focal_length = w / (2 * math.tan(math.radians(fov/2)))
        
        # Camera intrinsic matrix
        K = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ])
        
        # Rotation matrix (around Y-axis for right-side view)
        Ry = np.array([
            [math.cos(math.radians(angle)), 0, math.sin(math.radians(angle))],
            [0, 1, 0],
            [-math.sin(math.radians(angle)), 0, math.cos(math.radians(angle))]
        ])
        
        # Translation vector
        t = np.array([[X], [0], [Z]])
        
        # Full camera pose
        RT = np.hstack((Ry, t))
        
        # Projection matrix
        P = K @ RT
        
        # Define 3D coordinates of port corners (centered at origin)
        half_size = self.OUTER_SQUARE_SIZE / 2
        object_points = np.array([
            [-half_size, -half_size, 0],  # top-left
            [half_size, -half_size, 0],   # top-right
            [half_size, half_size, 0],    # bottom-right
            [-half_size, half_size, 0],   # bottom-left
        ], dtype=np.float32)
        
        # Project 3D points to 2D
        image_points = []
        for point in object_points:
            # Convert to homogeneous coordinates
# Convert to homogeneous coordinates
            point_h = np.append(point, 1)
            # Project point
            projected = P @ point_h
            # Convert to inhomogeneous coordinates
            projected = projected / projected[2]
            image_points.append(projected[:2])
        
        image_points = np.array(image_points, dtype=np.float32)
        
        # Get source points (original image corners)
        src_points = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)
        
        # Compute homography
        H, _ = cv2.findHomography(src_points, image_points)
        
        # Apply perspective transform
        transformed = cv2.warpPerspective(image, H, (w, h))
        
        # Draw guidelines to verify perspective
        if self.DEBUG:
            debug_img = transformed.copy()
            # Draw vertical and horizontal lines
            cv2.line(debug_img, 
                    tuple(image_points[0].astype(int)), 
                    tuple(image_points[1].astype(int)), 
                    (0, 255, 0), 2)
            cv2.line(debug_img, 
                    tuple(image_points[1].astype(int)), 
                    tuple(image_points[2].astype(int)), 
                    (0, 255, 0), 2)
            cv2.line(debug_img, 
                    tuple(image_points[2].astype(int)), 
                    tuple(image_points[3].astype(int)), 
                    (0, 255, 0), 2)
            cv2.line(debug_img, 
                    tuple(image_points[3].astype(int)), 
                    tuple(image_points[0].astype(int)), 
                    (0, 255, 0), 2)
            cv2.imshow("Perspective Guidelines", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return transformed, H

    def simulate_camera_movement(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please apply perspective transform first")
            return

        try:
            # Simulate camera movement until the angle approaches 0
            current_view = self.processed_image
            current_angle = 22.5  # Initial angle from perspective transform
            frames = []

            # Use a small threshold to determine when the image is "straight"
            while abs(current_angle) > 0.1:
                # Reduce angle by a small increment
                angle_step = min(current_angle, 1.0)
                current_angle -= angle_step

                # Apply perspective transform
                current_view, _ = self._apply_perspective_transform(
                    self.original_image, 
                    max(current_angle, 0)  # Ensure non-negative angle
                )
                
                frames.append(current_view)
                
                # Update UI to show progressive movement
                self.display_image(current_view)
                self.master.update()  # Force UI update
            
            # Display final (straight) image
            if frames:
                self.display_image(frames[-1])
                messagebox.showinfo("Camera Movement", "Camera movement complete")

        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = SatellitePortDetectorUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
