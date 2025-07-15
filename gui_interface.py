
import tkinter as tk
from tkinter import ttk, font
import cv2
import threading
from PIL import Image, ImageTk
import numpy as np
import queue
import time
import torch
import speech_recognition as sr
import os
from datetime import datetime

# System Information
CURRENT_UTC_TIME = "2025-05-31 02:48:28"
CURRENT_USER = "mmprateek1"

# Initialize critical imports first
try:
    np.zeros(1)
except:
    import pip
    pip.main(['install', '--upgrade', 'numpy'])
    import numpy as np

from object_detection.detector import ObjectDetector
from depth_estimation.depth import DepthEstimator
from navigation.navigator import Navigator
from env_description.descriptor import EnvironmentDescriptor
from face_analysis.analyzer import FaceAnalyzer

class VisionAssistantGUI:
    def __init__(self):
        # Initialize Tkinter root
        self.root = tk.Tk()
        self.root.title("Vision Assistant")
        self.root.geometry("1200x800")
        
        # Initialize logs
        self.init_logs()
        
        # Configure theme
        self.root.configure(bg='black')
        
        # Initialize CUDA
        self.init_cuda()
        
        # Initialize modules
        self.init_modules()
        
        # Initialize variables
        self.init_variables()
        
        # Setup GUI components
        self.setup_gui()
        
        # Setup Speech
        self.setup_speech()
        
        # Setup Voice Recognition
        self.setup_voice_recognition()
        
        # Add voice recognition status check
        self.root.after(1000, self.check_voice_recognition_status)
        
        # Automatically start camera
        self.root.after(100, self.start_camera)

    def log(self, message):
        """Log messages with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        if hasattr(self, 'log_file') and self.log_file is not None:
            try:
                self.log_file.write(log_entry)
                self.log_file.flush()
            except Exception as e:
                print(f"Error writing to log file: {str(e)}")

    def init_logs(self):
        """Initialize logging"""
        try:
            self.log_file = open("vision_assistant.log", "a")
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            self.log(f"Vision Assistant started by {os.getenv('USER', CURRENT_USER)} at {current_time}")
        except Exception as e:
            print(f"Error initializing log file: {str(e)}")
            self.log_file = None

    def init_cuda(self):
        """Initialize CUDA if available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.init()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    def init_modules(self):
        """Initialize detection and analysis modules"""
        try:
            self.object_detector = ObjectDetector()
            self.depth_estimator = DepthEstimator()
            self.navigator = Navigator()
            self.env_descriptor = EnvironmentDescriptor()
            self.face_analyzer = FaceAnalyzer()
            print("All modules initialized successfully")
        except Exception as e:
            print(f"Error initializing modules: {str(e)}")
            raise

    def init_variables(self):
        """Initialize instance variables"""
        # Camera variables
        self.camera = None
        self.is_running = False
        self.current_mode = None
        
        # Navigation variables
        self.navigation_target = None
        self.navigation_active = False
        self.last_navigation_update = time.time()
        self.navigation_update_interval = 2.0
        
        # Frame processing variables
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_thread = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Voice recognition variables
        self.voice_commands_enabled = True
        self.last_voice_command = None
        self.voice_command_cooldown = 1.0

    def setup_gui(self):
        """Setup GUI components"""
        # Create main frames
        self.control_frame = tk.Frame(self.root, bg='black')
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        
        self.video_frame = tk.Frame(self.root, bg='black')
        self.video_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Create video label
        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Create status labels
        self.create_status_labels()
        
        # Create mode buttons
        self.create_mode_buttons()
        
        # Configure window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_status_labels(self):
        """Create status and information labels"""
        # Status label
        status_text = f"FPS: 0 | GPU: {'Enabled' if torch.cuda.is_available() else 'Disabled'}"
        self.status_label = tk.Label(self.video_frame,
                                   text=status_text,
                                   fg='white', bg='black',
                                   font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        # Description label
        description_font = font.Font(size=14)
        self.description_label = tk.Label(self.video_frame,
                                        text="Select a mode to begin\nSay 'Navigate to [object]' for navigation",
                                        fg='white', bg='black',
                                        font=description_font,
                                        wraplength=500)
        self.description_label.pack(pady=20)
        
        # Voice command indicator
        self.voice_indicator = tk.Label(self.control_frame,
                                      text="Voice Commands: Active",
                                      fg='#00FF00', bg='black',
                                      font=('Arial', 12))
        self.voice_indicator.pack(pady=10)
        
        # Navigation status
        self.navigation_status = tk.Label(self.control_frame,
                                        text="Navigation: Inactive",
                                        fg='white', bg='black',
                                        font=('Arial', 12))
        self.navigation_status.pack(pady=10)
        
        # Debug label
        self.debug_label = tk.Label(self.control_frame,
                                  text="Debug Info: Ready",
                                  fg='white', bg='black',
                                  font=('Arial', 10),
                                  wraplength=200)
        self.debug_label.pack(pady=5)

    def create_mode_buttons(self):
        """Create mode selection buttons"""
        button_font = font.Font(size=16, weight='bold')
        modes = [
            ("Object Detection", "Detects and identifies objects in view"),
            ("Depth Estimation", "Shows distance to objects using colors"),
            ("Navigation Assistant", "Say 'Navigate to [object]' for guidance"),
            ("Environment Description", "Describes the surroundings"),
            ("Face Analysis", "Detects and analyzes faces")
        ]
        
        for mode, description in modes:
            button_frame = tk.Frame(self.control_frame, bg='black', pady=10)
            button_frame.pack(fill=tk.X)
            
            btn = tk.Button(button_frame,
                          text=mode,
                          font=button_font,
                          bg='#00FF00',
                          fg='black',
                          height=2,
                          command=lambda m=mode: self.toggle_mode(m))
            btn.pack(fill=tk.X)
            
            desc_label = tk.Label(button_frame,
                                text=description,
                                fg='white', bg='black',
                                font=('Arial', 12))
            desc_label.pack()

    def start_camera(self):
        """Start camera automatically"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.speak("Error: Could not open camera")
                self.log("Failed to open camera")
                self.debug_label.config(text="Error: Camera failed to open")
                return

            # Optimize camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_running = True
            self.log("Camera started automatically")
            self.speak("Vision Assistant is ready. You can say commands like 'Navigate to chair' or 'Describe environment'")
            
            # Start frame processing thread
            self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.processing_thread.start()
            
            # Start frame update
            self.update_frame()

        except Exception as e:
            self.log(f"Camera initialization error: {str(e)}")
            self.speak("Error initializing camera")
            self.debug_label.config(text=f"Camera Error: {str(e)}")

    def setup_speech(self):
        """Initialize text-to-speech"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.log("Text-to-speech initialized successfully")
        except Exception as e:
            self.log(f"Text-to-speech not available: {str(e)}")
            self.engine = None

    def setup_voice_recognition(self):
        """Initialize voice recognition"""
        try:
            self.recognizer = sr.Recognizer()
            self.is_running = True
            self.voice_command_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
            self.voice_command_thread.start()
            self.log("Voice recognition initialized successfully")
        except Exception as e:
            self.log(f"Error initializing voice recognition: {str(e)}")

    def check_voice_recognition_status(self):
        """Check if voice recognition is working"""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                self.log("Microphone check successful")
                self.voice_indicator.config(
                    text="Voice Commands: Active",
                    fg='#00FF00'
                )
                return True
        except Exception as e:
            self.log(f"Microphone check failed: {str(e)}")
            self.voice_indicator.config(
                text="Voice Commands: Error - Check Microphone",
                fg='#FF0000'
            )
            return False

    def speak(self, text):
        """Speak text using text-to-speech"""
        if self.engine and text != getattr(self, 'last_spoken_text', None):
            self.last_spoken_text = text
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        """Thread for text-to-speech processing"""
        try:
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            self.log(f"Speech error: {str(e)}")

    def toggle_mode(self, mode):
        """Toggle between different operation modes"""
        try:
            if self.current_mode != mode:
                self.current_mode = mode
                if mode == "Navigation Assistant":
                    self.navigation_active = True
                    self.speak("Navigation Assistant activated. Say 'Navigate to' followed by an object name")
                    self.navigation_status.config(
                        text="Navigation: Active - Waiting for target",
                        fg='#00FF00'
                    )
                    self.debug_label.config(text="Debug: Navigation mode activated")
                else:
                    self.navigation_active = False
                    self.navigation_target = None
                    self.speak(f"Selected mode: {mode}")
                    self.navigation_status.config(
                        text="Navigation: Inactive",
                        fg='white'
                    )
                self.log(f"Mode changed to: {mode}")
            else:
                self.speak(f"Mode {mode} is already active")
        except Exception as e:
            self.log(f"Mode toggle error: {str(e)}")
            self.debug_label.config(text=f"Error: {str(e)}")

    def listen_for_commands(self):
        """Listen for voice commands"""
        while self.is_running:
            try:
                if not self.voice_commands_enabled:
                    time.sleep(0.1)
                    continue

                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        command = self.recognizer.recognize_google(audio).lower()
                        self.log(f"Recognized command: {command}")
                        current_time = time.time()

                        if (self.last_voice_command is None or 
                            current_time - self.last_voice_command >= self.voice_command_cooldown):
                            self.process_voice_command(command)
                            self.last_voice_command = current_time

                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self.log(f"Speech recognition error: {str(e)}")

            except Exception as e:
                self.log(f"Voice command error: {str(e)}")
                self.debug_label.config(text=f"Voice Error: {str(e)}")
                time.sleep(0.1)

    def process_voice_command(self, command):
        """Process recognized voice commands"""
        if "navigate to" in command:
            target = command.split("navigate to")[1].strip()
            self.activate_navigation(target)
        elif "stop navigation" in command:
            self.stop_navigation()
        elif "pause navigation" in command:
            self.pause_navigation()
        elif "resume navigation" in command:
            self.resume_navigation()

    def activate_navigation(self, target):
        """Activate navigation to specified target"""
        self.navigation_target = target
        self.navigation_active = True
        self.current_mode = "Navigation Assistant"
        self.speak(f"Navigating to {target}")
        self.log(f"Navigation started to target: {target}")
        self.navigator.set_target(target)
        
        self.root.after(0, lambda: self.navigation_status.config(
            text=f"Navigation: Active - Target: {target}",
            fg='#00FF00'
        ))
        self.debug_label.config(text=f"Debug: Navigation target set to {target}")

    def stop_navigation(self):
        """Stop navigation"""
        self.navigation_active = False
        self.navigation_target = None
        self.navigator.target_object = None
        self.speak("Navigation stopped")
        self.log("Navigation stopped")
        self.navigation_status.config(
            text="Navigation: Inactive",
            fg='white'
        )

    def pause_navigation(self):
        """Pause navigation"""
        self.navigation_active = False
        self.speak("Navigation paused")
        self.log("Navigation paused")
        self.navigation_status.config(
            text="Navigation: Paused",
            fg='yellow'
        )

    def resume_navigation(self):
        """Resume navigation"""
        if self.navigation_target:
            self.navigation_active = True
            self.speak(f"Resuming navigation to {self.navigation_target}")
            self.log(f"Navigation resumed to: {self.navigation_target}")
            self.navigation_status.config(
                text=f"Navigation: Active - Target: {self.navigation_target}",
                fg='#00FF00'
            )

    def process_frames(self):
        """Process camera frames"""
        while self.is_running:
            try:
                if not self.camera or not self.camera.isOpened():
                    time.sleep(0.001)
                    continue
                
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    continue
                
                # Ensure frame is contiguous
                frame = np.ascontiguousarray(frame)
                
                # Process frame
                processed_frame, description = self.process_frame(frame)
                if processed_frame is not None:
                    self.frame_queue.put((processed_frame, description), block=False)
                
                # Update FPS
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time
                self.frame_count += 1
                
            except queue.Full:
                continue
            except Exception as e:
                self.log(f"Frame processing error: {str(e)}")
                self.debug_label.config(text=f"Processing Error: {str(e)}")
                time.sleep(0.001)

    def process_frame(self, frame):
        """Process frame based on current mode"""
        if not self.current_mode:
            return frame, "No mode selected"
            
        try:
            if frame is None:
                return None, "No frame available"
            
            if self.current_mode == "Object Detection":
                objects, frame = self.object_detector.detect(frame)
                if objects:
                    description = f"Detected {len(objects)} objects: " + \
                                ", ".join([f"{obj['class']} ({obj['confidence']:.2f})" for obj in objects])
                else:
                    description = "No objects detected"
                    
            elif self.current_mode == "Depth Estimation":
                frame = self.depth_estimator.estimate(frame)
                description = "Brighter colors indicate closer objects"
                
            elif self.current_mode == "Navigation Assistant":
                objects, annotated_frame = self.object_detector.detect(frame)
                depth_map = self.depth_estimator.estimate(frame)
                
                if self.navigation_active and self.navigation_target:
                    current_time = time.time()
                    if current_time - self.last_navigation_update >= self.navigation_update_interval:
                        guidance = self.navigator.get_guidance(objects, depth_map)
                        frame = self.navigator.visualize_path(annotated_frame, guidance)
                        description = guidance['recommendation']
                        self.last_navigation_update = current_time
                        self.debug_label.config(
                            text=f"Nav Debug: Target found={guidance.get('path_found', False)}"
                        )
                    else:
                        frame = annotated_frame
                        description = self.last_description if hasattr(self, 'last_description') else "Navigating..."
                else:
                    description = "Say 'Navigate to' followed by an object name"
                    frame = annotated_frame
                    
            elif self.current_mode == "Environment Description":
                objects, _ = self.object_detector.detect(frame)
                depth_map = self.depth_estimator.estimate(frame)
                env_info = self.env_descriptor.analyze(frame, objects, depth_map)
                description = env_info['description']
                
            elif self.current_mode == "Face Analysis":
                faces = self.face_analyzer.analyze(frame)
                if faces:
                    description = f"Detected {len(faces)} faces. " + \
                                ", ".join([f"Face at {face['position']}" for face in faces])
                else:
                    description = "No faces detected"
                
            return frame, description
                
        except Exception as e:
            self.log(f"Error in {self.current_mode}: {str(e)}")
            self.debug_label.config(text=f"Mode Error: {str(e)}")
            return frame, f"Error in {self.current_mode}: {str(e)}"

    def update_frame(self):
        """Update frame display"""
        if self.is_running:
            try:
                if not self.frame_queue.empty():
                    processed_frame, description = self.frame_queue.get_nowait()
                    
                    # Update description
                    if description != getattr(self, 'last_description', None):
                        self.description_label.config(text=description)
                        self.last_description = description
                        if self.current_mode == "Navigation Assistant" and self.navigation_active:
                            self.speak(description)
                        elif self.current_mode != "Navigation Assistant":
                            self.speak(description)
                    
                    # Update status
                    self.status_label.config(
                        text=f"FPS: {self.fps} | GPU: {'Enabled' if torch.cuda.is_available() else 'Disabled'}"
                    )
                    
                    # Update voice command indicator
                    self.voice_indicator.config(
                        text=f"Voice Commands: {'Active' if self.voice_commands_enabled else 'Paused'}",
                        fg='#00FF00' if self.voice_commands_enabled else '#FF0000'
                    )
                    
                    # Update navigation status
                    if self.navigation_active:
                        nav_text = f"Navigation: Active - Target: {self.navigation_target}"
                        nav_color = '#00FF00'
                    else:
                        nav_text = "Navigation: Inactive"
                        nav_color = 'white'
                    self.navigation_status.config(text=nav_text, fg=nav_color)
                    
                    # Update frame display
                    if processed_frame is not None:
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        self.video_label.config(image=frame_tk)
                        self.video_label.image = frame_tk
                    
            except queue.Empty:
                pass
            except Exception as e:
                self.log(f"Frame update error: {str(e)}")
                self.debug_label.config(text=f"Update Error: {str(e)}")
            
            # Schedule next update
            self.root.after(16, self.update_frame)

    def on_closing(self):
        """Handle window closing event"""
        self.cleanup()
        self.root.destroy()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'log_file') and self.log_file is not None:
            try:
                self.log("Application cleanup completed")
                self.log_file.close()
            except:
                print("Error closing log file")
            
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except Exception as e:
            self.log(f"Application error: {str(e)}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    app = VisionAssistantGUI()
    app.run()