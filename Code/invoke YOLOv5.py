import sys
sys.path.append("D:/learning things/fengru competition/Aero-engine defect-detect/yolov5/yolov5-master")
import detect

def detect_with_yolov5(weights, source, data, name):
    detect.detect(weights, source, data, name)

if __name__ == '__main__':
    weights = "for_wafers.pt"
    source = "D:/learning things/summer exercitation/dataset/wafers/From_ji_1/images/test"
    data = "D:/learning things/fengru competition/Aero-engine defect-detect/yolov5/yolov5-master/data/wafers.yaml"
    name = "myexp"
    detect_with_yolov5(weights, source, data, name)