import cv2
import os

def process_video(video_path, output_path, frame_interval):
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return []
    
    # Lấy thông tin video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nThông tin video:")
    print(f"  - Tổng số frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Kích thước: {width}x{height}")
    print(f"  - Detect mỗi {frame_interval} frames")
    
    # Tạo VideoWriter để ghi video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    predictions = []
    frame_count = 0
    current_prediction = None
    
    print("Đang xử lý video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_path, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)

    cap.release()
    out.release()
        
        
        
        
if __name__ == "__main__":
    video_path = "video_output/K01_V001_001.mp4"
    output_path = "keyframes/V001_001"
    process_video(video_path, output_path, frame_interval=30)
        