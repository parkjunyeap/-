import sys
import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, Listbox, LabelFrame, Entry, Button
from PIL import Image, ImageTk
from pathlib import Path
import winsound  # 비프음을 위한 모듈 
import threading # 비프음을 쓰레드로 들리게 함.
import pygame

# Pygame 초기화
pygame.mixer.init()

# YOLOv7 코드베이스 경로 추가
FILE = os.path.abspath(__file__) #현재 실행 중인 파일의 절대 경로를 얻습니다. __file__은 현재 파일의 경로를 나타내며, os.path.abspath는 이를 절대 경로로 변환
ROOT = os.path.dirname(FILE) #절대 경로에서 디렉토리 경로를 추출합니다. 즉, 현재 파일이 위치한 디렉토리 경로를 얻습니다.
if ROOT not in sys.path: #경로가 파이썬의 모듈 검색 경로 목록에 없는지 확인합니다.
    sys.path.append(ROOT)

from models.experimental import attempt_load # 모듈로드
from utils.general import non_max_suppression, scale_coords

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # cuda0 
# 모델을 CPU로 로드
model = attempt_load('yolov7.pt', map_location=device)
model.to(device).eval()

def preprocess(image, img_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, _, _ = letterbox(image, new_shape=img_size)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float().to(device)
    image /= 255.0
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    return image

def letterbox(img, new_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def detect_objects(image, model, device):
    # img = torch.from_numpy(image).to(device)  # 입력 이미지를 GPU로 이동
    img = preprocess(image, image.shape[1])
    # img = img.float() / 255.0  # 이미지 정규화

    # if len(img.shape) == 3:
    #     img = img.permute(2, 0, 1).unsqueeze(0)  # [height, width, channels] -> [channels, height, width] -> [1, channels, height, width]

    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # pred 텐서를 CPU로 이동하여 non_max_suppression 실행
    pred = pred.cpu()
    results = non_max_suppression(pred, 0.25, 0.45, classes=[0], agnostic=False)
    
    return results

def postprocess(results, img_shape, frame_shape, conf_threshold=0.5):
    boxes = []
    if results is None or len(results) == 0:
        return boxes
    
    for det in results:
        if len(det):
            # det[:, :4] = scale_coords(img_shape, det[:, :4], frame_shape).round()
            for *xyxy, conf, cls in reversed(det):
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                boxes.append((bbox, f'{model.names[int(cls)]} {conf:.2f}'))
    return boxes

def async_beep():
    def play_beep():
        pygame.mixer.music.load('beep.wav')
        pygame.mixer.music.play()
    
    threading.Thread(target=play_beep).start()

    # 비프음 

def draw_boxes(image, boxes, roi):
    roi_x, roi_y, roi_w, roi_h = roi
    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    for bbox, label in boxes:
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        color = (0, 255, 0)
        if not (roi_x < p1[0] < roi_x + roi_w and roi_y < p1[1] < roi_y + roi_h and
                roi_x < p2[0] < roi_x + roi_w and roi_y < p2[1] < roi_y + roi_h):
            color = (0, 0, 255) 
            # cv2.putText(image, 'Object out of ROI', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            async_beep()  # 비프음을 비동기적으로 재생  # 객체가 ROI를 벗어나면 비프음을 재생 # 1000 Hz로 500 ms 사운드
        cv2.rectangle(image, p1, p2, color, 2, 1)
    return image

class CameraApp:
    def __init__(self, root, camera_indices):
        self.root = root
        self.camera_indices = camera_indices
        self.cameras = []
        self.rois = []
        self.selected_camera = None
        self.frame = None
        self.img_size = 640
        
        self.root.title("Multiple Camera Viewer")
        self.root.geometry("660x600")
        
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.camera_buttons = []
        for i, cam_index in enumerate(camera_indices):
            button = tk.Button(self.top_frame, text=f"Camera {i+1}", command=lambda i=i: self.select_camera(i))
            button.pack(side=tk.LEFT, padx=(10 if i == 0 else 5, 5))
            self.camera_buttons.append(button)
        
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, width=640, height=640)
        self.canvas.pack()

        self.list_frame = None
        self.file_frame = None
        self.path_frame = None

        self.load_cameras()

    # 실행시 카메라를 불러옴
    def load_cameras(self):
        for cam_index in self.camera_indices:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                # messagebox.showerror("Error", f"카메라 {cam_index}를 불러올 수 없습니다.")
                self.cameras.append(None)
            else:
                self.cameras.append(cap)
                ret, frame = cap.read()
                if ret:
                    roi = cv2.selectROI("Select ROI", cv2.flip(frame, 1), fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Select ROI")
                    self.rois.append(roi)
                else:
                    self.rois.append(None)
    
    # 카메라 선택
    def select_camera(self, camera_index):
        if self.selected_camera == camera_index:
            pass
        else:
            if camera_index != 3:
                self.hide_listbox()
                self.hide_file_buttons()
            else:
                self.show_listbox()
                self.show_file_buttons()
            self.selected_camera = camera_index
            self.show_frame()

    def show_frame(self):
        self.canvas_frame.pack()
        if self.selected_camera is None or self.cameras[self.selected_camera] is None:
            self.canvas.delete("all")
            self.canvas.create_text(320, 320, text="{0}번 카메라를 불러올 수 없습니다".format(self.selected_camera+1), fill="red", font=('Helvetica', 20))
            if self.selected_camera == 3:
                self.canvas_frame.pack_forget()
            return
        
        cap = self.cameras[self.selected_camera]
        roi = self.rois[self.selected_camera]
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        results = detect_objects(frame, model, device)
        boxes = postprocess(results, (self.img_size, self.img_size), frame.shape)
        output_frame = draw_boxes(frame, boxes, roi)

        image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

        self.root.after(10, self.show_frame)

    def play_video(self, file):
        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        while cap.isOpened:
            ret, frame = cap.read()

            results = detect_objects(frame, model, device)
            boxes = postprocess(results, (self.img_size, self.img_size), frame.shape)
            frame = draw_boxes(frame, boxes, roi)

            cv2.imshow("{0}".format(file), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def show_listbox(self):
        if self.list_frame is None:
            self.list_frame = tk.Frame(self.root)
            self.list_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)
            
            self.scrollbar = tk.Scrollbar(self.list_frame)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.list_file = Listbox(self.list_frame, selectmode="extended", height=15, yscrollcommand=self.scrollbar.set)
            self.list_file.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.list_file.bind("<Double-Button-1>", self.on_listbox_double_click)  # 더블 클릭 이벤트 바인딩
            self.scrollbar.config(command=self.list_file.yview)

    def hide_listbox(self):
        if self.list_frame is not None:
            self.list_frame.pack_forget()
            self.list_frame = None

    def show_file_buttons(self):
        if self.file_frame is None:
            self.file_frame = tk.Frame(self.root)
            self.file_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
            
            self.btn_add_file = Button(self.file_frame, padx=5, pady=5, width=12, text="파일추가", command=self.add_file)
            self.btn_add_file.pack(side="left")

            self.btn_del_file = Button(self.file_frame, padx=5, pady=5, width=12, text="선택삭제", command=self.del_file)
            self.btn_del_file.pack(side="right")
    
    def hide_file_buttons(self):
        if self.file_frame is not None:
            self.file_frame.pack_forget()
            self.file_frame = None

    # 리스트에서 선택한 비디오 파일 재생
    def on_listbox_double_click(self, event):
        selected_file = self.list_file.get(self.list_file.curselection())
        self.play_video(file=selected_file)

    def add_file(self):
        files = filedialog.askopenfilenames(title="비디오 파일을 선택하세요", \
            filetypes=(("MP4 파일", "*.mp4"), ("모든 파일", "*.*")), \
            initialdir=os.path.expanduser('~'))  # 사용자 홈 디렉토리를 기본 경로로 설정
    
        # 사용자가 선택한 파일 목록
        for file in files:
            self.list_file.insert('end', file)

    def del_file(self):
        for index in reversed(self.list_file.curselection()):
            self.list_file.delete(index)

    def browse_dest_path(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, camera_indices=[0, 1, 2, 3])
    root.mainloop()