import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
from collections import deque

# Load trained model
with open('stretching.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

CONFIDENCE_THRESHOLD = 0.6  
BUFFER_SIZE = 3  
POSE_COUNT_REQUIRED = 15  
prediction_buffer = deque(maxlen=BUFFER_SIZE)

# Define the required sequence
POSE_SEQUENCE = [
    "t-pose",
    "Stretch-left",
    "Stretch-right",
    "stretch Left hand up",
    "stretch Right hand up"
]

class PosePopup:
    skip_count = 0  

    def __init__(self, root):
        self.root = root
        self.root.title("Time's Up!")  
        self.root.attributes("-fullscreen", True)  
        self.root.attributes("-topmost", True)  # Always stay on top
        self.root.lift()  # Bring window to front
        #self.root.focus_force()

        
        self.bg_image = Image.open("assets/background.png")  
        self.bg_image = self.bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        
        self.canvas = tk.Canvas(root, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)
        self.canvas.pack(fill="both", expand=True)

      
        self.current_pose_index = 0
        self.pose_count = 0
        self.is_running = True
        self.pose_verified = False

        
        self.gif_frames = []
        self.durations = []
        self.current_frame_index = 0
        self.after_id = None

        #(Always Cyan)
        self.pose_status_label = tk.Label(root, text=f"Do a {POSE_SEQUENCE[self.current_pose_index]}: 0/{POSE_COUNT_REQUIRED}", 
                                          font=("Arial", 34, "bold"), fg="#00c4cc", bg="white",
                                          padx=20, pady=10)
        self.pose_status_label.place(relx=0.5, rely=0.12, anchor="center")  

        # Video feed
        self.video_label = tk.Label(root, bg="white")
        self.video_label.place(relx=0.25, rely=0.6, anchor="center")  

        # GIF Label
        self.pose_gif_label = tk.Label(root, bg="white")
        self.pose_gif_label.place(relx=0.765, rely=0.6, anchor="center")  # Positioned on the right side
        self.update_pose_gif()  # Load the first pose GIF

        # Skip Button (Top-Right)
        skip_text = f"Skip ({PosePopup.skip_count + 1}/3)" if PosePopup.skip_count < 3 else "Skip (3/3)"
        self.skip_button = tk.Button(root, text=skip_text, font=("Arial", 14, "bold"), command=self.close_popup,
                                     bg="#00c4cc", fg="white", bd=0, highlightthickness=0, relief="flat",
                                     activebackground="#00a0a3", activeforeground="white")  
        self.skip_button.place(relx=0.99, rely=0.0095, anchor="ne")  
        # Hide skip button 
        if PosePopup.skip_count >= 3:
            self.skip_button.place_forget()

        # Exit Button
        self.exit_button = tk.Button(root, text="🎉 Exit 🎉", font=("Arial", 18, "bold"), command=self.close_popup,
                                    bg="#2e7a89", fg="white", padx=20, pady=8, 
                                    bd=0, highlightthickness=0, relief="flat", 
                                    cursor="hand2", activebackground="#256974")  
        self.exit_button.place(relx=0.5, rely=0.85, anchor="center")  
        self.exit_button.place_forget()  

       
        self.pose_thread = Thread(target=self.run_pose_detection, daemon=True)
        self.pose_thread.start()

    def run_pose_detection(self):
        """Runs pose detection and updates the video feed in the window."""
        self.cap = cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Convert BGR to RGB for Mediapipe processing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize the frame (Make it 70% of original size)
                scale_percent = 70  
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                results = pose.process(image)
                image.flags.writeable = True

                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                    try:
                        # Extract pose landmarks
                        pose_landmarks = results.pose_landmarks.landmark
                        row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())

                        # Convert to DataFrame for prediction
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]

                        # Only accept predictions above confidence threshold
                        max_prob = np.max(body_language_prob)
                        if max_prob >= CONFIDENCE_THRESHOLD:
                            prediction_buffer.append(body_language_class)

                        if len(prediction_buffer) == BUFFER_SIZE:
                            final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                        else:
                            final_prediction = "Waiting..."

                        print(f"Detected Pose: {final_prediction}")

                 
                        if final_prediction == POSE_SEQUENCE[self.current_pose_index]:
                            self.pose_count += 1
                            self.pose_status_label.config(
                                text=f"{POSE_SEQUENCE[self.current_pose_index]}: {self.pose_count}/{POSE_COUNT_REQUIRED}",
                                fg="#00c4cc"  
                            )


                        if self.pose_count >= POSE_COUNT_REQUIRED:
                            if self.current_pose_index < len(POSE_SEQUENCE) - 1:
                                self.current_pose_index += 1
                                self.pose_count = 0
                                self.pose_status_label.config(
                                    text=f"Do a {POSE_SEQUENCE[self.current_pose_index]}: 0/{POSE_COUNT_REQUIRED}",
                                    fg="#00c4cc"
                                )
                                self.update_pose_gif()  # Change GIF
                            else:
                           
                                self.pose_status_label.config(
                                    text="🎉 All Poses Complete! 🎉",
                                    fg="#00c4cc"
                                )
                                self.exit_button.place(relx=0.5, rely=0.85, anchor="center")  
                                self.pose_verified = True  

                    except Exception as e:
                        print(f"Prediction error: {e}")

                # Convert frame for Tkinter
                image = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=image)

                # Show camera feed in the window
                if self.is_running:
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

        self.cap.release()
        cv2.destroyAllWindows()

    def update_pose_gif(self):
        """Updates the GIF when the pose changes."""
        gif_path = f"assets/{POSE_SEQUENCE[self.current_pose_index]}.gif"
        try:
            
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None

            
            gif = Image.open(gif_path)
            self.gif_frames = []
            self.durations = []
            for frame_number in range(gif.n_frames):
                gif.seek(frame_number)
                # Resize the frame
                resized_frame = gif.copy().resize((300, 300), Image.LANCZOS)
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(resized_frame)
                self.gif_frames.append(photo)
                
                self.durations.append(gif.info.get('duration', 100))
            
            self.current_frame_index = 0
            # Start animation
            self.animate_gif()
        except Exception as e:
            print(f"Error loading GIF: {e}")

    def animate_gif(self):
        """Animates the loaded GIF frames."""
        if not self.is_running:
            return  
        if self.current_frame_index >= len(self.gif_frames):
            self.current_frame_index = 0  
        # Update the image
        self.pose_gif_label.config(image=self.gif_frames[self.current_frame_index])
        
        delay = self.durations[self.current_frame_index]
        self.current_frame_index += 1
        self.after_id = self.root.after(delay, self.animate_gif)

    def close_popup(self):
       
        if not self.exit_button.winfo_ismapped():
            PosePopup.skip_count += 1  
        else:
            PosePopup.skip_count = 0  
        self.is_running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

def show_popup():
    root = tk.Tk()
    app = PosePopup(root)
    root.mainloop()