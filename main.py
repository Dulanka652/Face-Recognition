import cv2
import numpy as np
from insightface.app import FaceAnalysis
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class FaceVerifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Face Verification System")
        self.geometry("800x650")
        
        # InsightFace Model Loading
        print("Loading InsightFace Model (buffalo_l)...")
        self.fa = FaceAnalysis(name="buffalo_l")
        self.fa.prepare(ctx_id=0, det_size=(640, 640))
        print("Model Loaded Successfully!")
        
        self.path1 = None
        self.path2 = None
        
        self.create_widgets()

    def create_widgets(self):
        self.title_lbl = ctk.CTkLabel(self, text="AI Face Verification & Matching", font=("Poppins", 24, "bold"))
        self.title_lbl.pack(pady=20)

        self.frame = ctk.CTkFrame(self)
        self.frame.pack(pady=10, fill="both", expand=True, padx=30)
        

        self.left_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.left_frame.grid(row=0, column=0, padx=40, pady=20)
        self.btn_select1 = ctk.CTkButton(self.left_frame, text="Choose Person 1", command=self.select_image1, fg_color="#10b981")
        self.btn_select1.pack(pady=10)
        self.lbl_img1 = ctk.CTkLabel(self.left_frame, text="No Image Selected", width=250, height=250, fg_color="#1e293b", corner_radius=12)
        self.lbl_img1.pack()

      
        self.right_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=40, pady=20)
        self.btn_select2 = ctk.CTkButton(self.right_frame, text="Choose Person 2", command=self.select_image2, fg_color="#10b981")
        self.btn_select2.pack(pady=10)
        self.lbl_img2 = ctk.CTkLabel(self.right_frame, text="No Image Selected", width=250, height=250, fg_color="#1e293b", corner_radius=12)
        self.lbl_img2.pack()
        
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        self.btn_verify = ctk.CTkButton(self, text="Verify Faces", command=self.verify_faces, font=("Arial", 16, "bold"), height=45, width=200)
        self.btn_verify.pack(pady=20)

        self.lbl_result = ctk.CTkLabel(self, text="Result: Awaiting Input...", font=("Arial", 18, "bold"), text_color="#94a3b8")
        self.lbl_result.pack(pady=10)

    def load_and_display_image(self, lbl_widget):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((250, 250))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))
            lbl_widget.configure(image=ctk_img, text="")
            return file_path
        return None

    def select_image1(self):
        self.path1 = self.load_and_display_image(self.lbl_img1)

    def select_image2(self):
        self.path2 = self.load_and_display_image(self.lbl_img2)

    def verify_faces(self):
        if not self.path1 or not self.path2:
            messagebox.showwarning("Warning", "Please select both images!")
            return

        img1 = cv2.imread(self.path1)
        img2 = cv2.imread(self.path2)

        if img1 is None or img2 is None:
            messagebox.showerror("Error", "Check file paths or file names (Use simple English).")
            return

        facesA = self.fa.get(img1)
        facesB = self.fa.get(img2)

        if len(facesA) == 0 or len(facesB) == 0:
            messagebox.showerror("Error", "No face detected in one or both images!")
            return

    
        code1 = facesA[0].normed_embedding
        code2 = facesB[0].normed_embedding
        
 
        cosine_sim = np.dot(code1, code2) / (np.linalg.norm(code1) * np.linalg.norm(code2))
    
        similarity = (cosine_sim + 1) * 50

        if cosine_sim > 0.3:
            self.lbl_result.configure(text=f"MATCH FOUND! ({similarity:.2f}% Similarity)", text_color="#10b981")
        else:
            self.lbl_result.configure(text=f"NOT A MATCH! ({similarity:.2f}% Similarity)", text_color="#ef4444")

if __name__ == "__main__":
    app = FaceVerifierApp()
    app.mainloop()