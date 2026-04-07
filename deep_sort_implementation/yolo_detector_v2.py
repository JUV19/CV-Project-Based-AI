import torch

class YoloDetector:
    def __init__(self, confidence):
        # For custom models trained with YOLOv5 architecture/repo
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="models/weights.pt",
            force_reload=False
        )

        self.model.conf = confidence
        self.confidence = confidence

        # All class names from the model
        # Example: {0: 'person', 1: 'helmet', 2: 'vest', 3: 'vehicle'}
        self.classList = list(self.model.names.values())

    def detect(self, image):
        results = self.model(image)
        return self.make_detections(results)

    def make_detections(self, results):
        detections = []

        # YOLOv5 torch.hub format per detection:
        # [x1, y1, x2, y2, conf, cls]
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, class_number = det.tolist()

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_number = int(class_number)
            class_name = self.model.names[class_number]
            conf = float(conf)

            # DeepSORT expects: ([x, y, w, h], confidence, class_name)
            detections.append(([x1, y1, w, h], conf, class_name))

        return detections