from ultralytics import YOLO

model = YOLO('best.pt')

results = model(
    'test_video.mp4',
    save=True,
    conf=0.4,
    project='outputs',
    name='video_detection'
)

print("Done! Check outputs/video_detection/ for the annotated video.")