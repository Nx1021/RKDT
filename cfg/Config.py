# from utils.yaml import yaml_load
# class Config():
#     def __init__(self, yaml) -> None =
#                 #Ultralytics YOLO 🚀, AGPL-3.0 license
#                 #Default training settings and hyperparameters for medium-augmentation COCO training
#         args = yaml_load(yaml)

#         self.task           = args["task"]          #YOLO task, i.e. detect, segment, classify, pose
#         self.mode           = args["mode"]          #YOLO mode, i.e. train, val, predict, export, track, benchmark

#         #Train settings -------------------------------------------------------------------------------------------------------
#         self.model          = arg[""] #path to model file, i.e. yolov8n.pt, yolov8n.yaml
#         self.data           = arg[""] #path to data file, i.e. coco128.yaml
#         self.epochs         = arg[""]     #number of epochs to train for
#         self.patience       = arg[""]    #epochs to wait for no observable improvement for early stopping of training
#         self.batch          = arg[""]    #number of images per batch (-1 for AutoBatch)
#         self.imgsz          = arg[""]     #size of input images as integer or w,h
#         self.save           = arg[""]      #save train checkpoints and predict results
#         self.save_period    = arg[""]   #Save checkpoint every x epochs (disabled if < 1)
#         self.cache          = arg[""]       #True/ram, disk or False. Use cache for data loading
#         self.device         = arg[""] #device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
#         self.workers        = arg[""]   #number of worker threads for data loading (per RANK if DDP)
#         self.project        = arg[""] #project name
#         self.name           = arg[""] #experiment name, results saved to 'project/name' directory
#         self.exist_ok       = arg[""]       #whether to overwrite existing experiment
#         self.pretrained     = arg[""]       #whether to use a pretrained model
#         self.optimizer      = arg[""]     #optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
#         self.verbose        = arg[""]      #whether to print verbose output
#         self.seed           = arg[""]   #random seed for reproducibility
#         self.deterministic  = arg[""]      #whether to enable deterministic mode
#         self.single_cls     = arg[""]       #train multi-class data as single-class
#         self.rect           = arg[""]       #rectangular training if mode='train' or rectangular validation if mode='val'
#         self.cos_lr         = arg[""]       #use cosine learning rate scheduler
#         self.close_mosaic   = arg[""]   #(int) disable mosaic augmentation for final epochs
#         self.resume         = arg[""]       #resume training from last checkpoint
#         self.amp            = arg[""]      #Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
#         self.fraction       = arg[""]       #dataset fraction to train on (default is 1.0, all images in train set)
#         self.profile        = arg[""]       #profile ONNX and TensorRT speeds during training for loggers
#         #Segmentation
#         self.overlap_mask   = arg[""]          #masks should overlap during training (segment train only)
#         self.mask_ratio     = arg[""]              #mask downsample ratio (segment train only)
#         #Classificationarg[""]
#         self.dropout        = arg[""]          #use dropout regularization (classify train only)

#         #Val/Test settings ----------------------------------------------------------------------------------------------------
#         self.val            = arg[""]         #validate/test during training
#         self.split          = arg[""]         #dataset split to use for validation, i.e. 'val', 'test' or 'train'
#         self.save_json      = arg[""]         #save results to JSON file
#         self.save_hybrid    = arg[""]         #save hybrid version of labels (labels + additional predictions)
#         self.conf           = arg[""]         #object confidence threshold for detection (default 0.25 predict, 0.001 val)
#         self.iou            = arg[""]         #intersection over union (IoU) threshold for NMS
#         self.max_det        = arg[""]         #maximum number of detections per image
#         self.half           = arg[""]         #use half precision (FP16)
#         self.dnn            = arg[""]         #use OpenCV DNN for ONNX inference
#         self.plots          = arg[""]         #save plots during train/val
#         #Prediction settings -------------------------------------------------------------------------------------------
#         self.source         = arg[""]         #source directory for images or videos
#         self.show           = arg[""]         #show results if possible
#         self.save_txt       = arg[""]         #save results as .txt file
#         self.save_conf      = arg[""]         #save results with confidence scores
#         self.save_crop      = arg[""]         #save cropped images with results
#         self.show_labels    = arg[""]        #show object labels in plots
#         self.show_conf      = arg[""]        #show object confidence scores in plots
#         self.vid_stride     = arg[""]         #video frame-rate stride
#         self.line_width     = arg[""]         #line width of the bounding boxes
#         self.visualize      = arg[""]         #visualize model features
#         self.augment        = arg[""]         #apply image augmentation to prediction sources
#         self.agnostic_nms   = arg[""]         #class-agnostic NMS
#         self.classes        = arg[""]         #filter results by class, i.e. class=0, or class=[0,2,3]
#         self.retina_masks   = arg[""]         #use high-resolution segmentation masks
#         self.boxes          = arg[""]        #Show boxes in segmentation predictions

#         #Export settings ------------------------------------------------------------------------------------------------------
#         self.format         = torchscript          #format to export to
#         self.keras          = False          #use Keras
#         self.optimize       = False          #TorchScript = optimize for mobile
#         self.int8           = False          #CoreML/TF INT8 quantization
#         self.dynamic        = False          #ONNX/TF/TensorRT = dynamic axes
#         self.simplify       = False          #ONNX = simplify model
#         self.opset          =          #ONNX = opset version (optional)
#         self.workspace      = 4          #TensorRT = workspace size (GB)
#         self.nms            = False          #CoreML = add NMS

#                 #Hyperparameters ------------------------------------------------------------------------------------------------------
#         self.lr0            = 0.01          #initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
#         self.lrf            = 0.01          #final learning rate (lr0 * lrf)
#         self.momentum       = 0.937          #SGD momentum/Adam beta1
#         self.weight_decay   = 0.0005          #optimizer weight decay 5e-4
#         self.warmup_epochs  = 3.0          #warmup epochs (fractions ok)
#         self.warmup_momentum = 0.8          #warmup initial momentum
#         self.warmup_bias_lr = 0.1          #warmup initial bias lr
#         self.box            = 7.5          #box loss gain
#         self.cls            = 0.5          #cls loss gain (scale with pixels)
#         self.dfl            = 1.5          #dfl loss gain
#         self.pose           = 12.0          #pose loss gain
#         self.kobj           = 1.0          #keypoint obj loss gain
#         self.label_smoothing = 0.0          #label smoothing (fraction)
#         self.nbs            = 64          #nominal batch size
#         self.hsv_h          = 0.015          #image HSV-Hue augmentation (fraction)
#         self.hsv_s          = 0.7          #image HSV-Saturation augmentation (fraction)
#         self.hsv_v          = 0.4          #image HSV-Value augmentation (fraction)
#         self.degrees        = 0.0          #image rotation (+/- deg)
#         self.translate      = 0.1          #image translation (+/- fraction)
#         self.scale          = 0.5          #image scale (+/- gain)
#         self.shear          = 0.0          #image shear (+/- deg)
#         self.perspective    = 0.0          #image perspective (+/- fraction), range 0-0.001
#         self.flipud         = 0.0          #image flip up-down (probability)
#         self.fliplr         = 0.5          #image flip left-right (probability)
#         self.mosaic         = 1.0          #image mosaic (probability)
#         self.mixup          = 0.0          #image mixup (probability)
#         self.copy_paste     = 0.0          #segment copy-paste (probability)

#                 #Custom config.yaml ---------------------------------------------------------------------------------------------------
#         self.cfg            =          #for overriding defaults.yaml

#                 #Debug, do not modify -------------------------------------------------------------------------------------------------
#         self.v5loader       = False          #use legacy YOLOv5 dataloader

#                 #Tracker settings ------------------------------------------------------------------------------------------------------
#         self.tracker        = botsort.yaml          #tracker type, ['botsort.yaml', 'bytetrack.yaml']
