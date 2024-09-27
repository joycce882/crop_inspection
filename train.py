from ultralytics import YOLO

model = YOLO(model='/root/yolov8/runs/classify/train2/weights/best.pt')
result = model.train(data = '/root/datasets/dataset',epochs=30,batch=256,device = '0',workers=8,optimizer = 'SGD'
                     ,lr0 = 0.001,cos_lr =True,patience = 5)