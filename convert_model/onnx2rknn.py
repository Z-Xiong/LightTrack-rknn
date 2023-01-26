import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import onnx
import onnxruntime
from PIL import Image
from onnxsim import simplify
import math

MODEL = 2
do_quantization = False
do_analysis = False
do_eval_perf = False
do_inference = False

if MODEL == 0:
    ONNX_MODEL = 'model/lighttrack_init.onnx'
    ONNXSIM_MODEL = 'model/lighttrack_init-sim.onnx'
    RKNN_MODEL = 'model/lighttrack_init.rknn'
    RKNN_MODEL_FLOAT32 = 'model/lighttrack_init_f32.rknn'
    RKNN_PRECOMPILE_MODEL = 'model/lighttrack_init.precompile.rknn'
    CRYPT_MODEL = 'model/lighttrack_init.crypt.rknn'
    IMG_FILE = 'dataset_init/000034-baby0.jpg'
    INPUTS = [[1, 3, 127, 127]]
    dataset = 'dataset_init.txt'
    dataset_aa = 'dataset_aa_init.txt'
    model_input="hq_cfg/init.json"
    data_input="hq_cfg/init.data"
    model_quantization_cfg="hq_cfg/init.quantization.cfg"

if MODEL == 1:
    ONNX_MODEL = 'model/lighttrack_backbone.onnx'
    ONNXSIM_MODEL = 'model/lighttrack_backbone-sim.onnx'
    RKNN_MODEL = 'model/lighttrack_backbone.rknn'
    RKNN_MODEL_FLOAT32 = 'model/lighttrack_backbone_f32.rknn'
    RKNN_PRECOMPILE_MODEL = 'model/lighttrack_backbone.precompile.rknn'
    CRYPT_MODEL = 'model/lighttrack_backbone.crypt.rknn'
    IMG_FILE = 'dataset_backbone/000034-baby0.jpg'
    INPUTS = [[1, 3, 288, 288]]
    dataset = 'dataset_backbone.txt'
    dataset_aa = 'dataset_aa_backbone.txt'
    model_input="hq_cfg/backbone.json"
    data_input="hq_cfg/backbone.data"
    model_quantization_cfg="hq_cfg/backbone.quantization.cfg"

if MODEL == 2:
    ONNX_MODEL = 'model/lighttrack_neck_head.onnx'
    ONNXSIM_MODEL = 'model/lighttrack_neck_head-sim.onnx'
    RKNN_MODEL = 'model/lighttrack_neck_head.rknn'
    RKNN_MODEL_FLOAT32 = 'model/lighttrack_neck_head_f32.rknn'
    RKNN_PRECOMPILE_MODEL = 'model/lighttrack_neck_head.precompile.rknn'
    CRYPT_MODEL = 'model/lighttrack_neck_head.crypt.rknn'
    IMG_FILE = "dataset_neckhead/zf_rknn.npy dataset_neckhead/xf_rknn.npy"    # (NHWC格式)
    INPUTS = [[1, 96, 8, 8], [1, 96, 18, 18]]
    dataset = 'dataset_neckhead.txt'
    dataset_aa = 'dataset_neckhead.txt'
    model_input="hq_cfg/neckhead.json"
    data_input="hq_cfg/neckhead.data"
    model_quantization_cfg="hq_cfg/neckhead.quantization.cfg"


def consine(file1, file2):
    f1 = open(file1, "r")
    f2 = open(file2, "r")

    data1 = f1.readlines()
    data2 = f2.readlines()
    
    f1.close()
    f2.close()

    sum_ = 0
    sqrt1 = 0
    sqrt2 = 0
    for i in range(len(data1)):
        sum_ += (float(data1[i]))*(float(data2[i]))
        sqrt1 += float(data1[i]) * float(data1[i]) 
        sqrt2 += float(data2[i]) * float(data2[i])
    consine = sum_ / (math.sqrt(sqrt1) * math.sqrt(sqrt2))
    return consine


def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)

# Preprocess anormalize the image
def preprocess(img_file, w, h):
    input_shape = (1, 3, w, h)
    img = Image.open(img_file).convert('RGB')
    # img = img.resize((w, h), Image.BILINEAR)
    # convert the input data into the float32 input
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    # mean_vec = np.array([0.485, 0.456, 0.406])
    # stddev_vec = np.array([0.229, 0.224, 0.225])
    mean_vec = np.array([123.675, 116.28, 103.53])
    stddev_vec = np.array([58.3942, 57.12, 57.3756])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (img_data[:, i, :, :] - mean_vec[i]) / stddev_vec[i]
    return norm_img_data.astype('float32'), np.array(img)


def onnx_predict(onnx_model, img_file, output_txt=False):
    if MODEL == 0 or MODEL==1:
        # Run the model on the backend
        session = onnxruntime.InferenceSession(onnx_model, None)

        # get the name of the first input of the model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        # print(len(session.get_outputs()))
        print('Input Name:', input_name)
        print('Output Name:', output_name)
        # 符合https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd 模型的输入要求
        input_data, raw_img = preprocess(img_file, INPUTS[0][2], INPUTS[0][3])
        print('输入图像大小：', input_data.shape)

        start = time.time()
        raw_result = session.run([], {input_name: input_data})   # 输入个是（NCHW）= （1，3，127, 127）
        end = time.time()
        print('推理时间：', end - start, 's')
        raw_result_np = np.array(raw_result[0])
        print("output shape is: ", raw_result_np.shape)
        raw_result_size = raw_result_np.shape[0] * raw_result_np.shape[1] * raw_result_np.shape[2] * raw_result_np.shape[3]

        if output_txt:   
            f = open("onnx_predict.txt", "w")
            for i in range(0, raw_result_np.shape[0]):
                for j in range(0, raw_result_np.shape[1]):
                    for k in range(0, raw_result_np.shape[2]):
                        for m in range(0, raw_result_np.shape[3]):
                            f.write(str(raw_result_np[i, j, k, m])+' ')
                        f.write('\n')
                    f.write('\n')
            f.close()
    else:
        # Run the model on the backend
        session = onnxruntime.InferenceSession(onnx_model, None)

        # get the name of the first input of the model
        input_name_1 = session.get_inputs()[0].name
        input_name_2 = session.get_inputs()[1].name
        output_name_1 = session.get_outputs()[0].name
        output_name_2 = session.get_outputs()[1].name
        # print(len(session.get_outputs()))
        print('Input Name 1:', input_name_1)
        print('Input Name 2:', input_name_2)
        print('Output Name 1:', output_name_1)
        print('Output Name 2:', output_name_2)
        input_data_1 = np.load(img_file.split(' ')[0])
        input_data_2 = np.load(img_file.split(' ')[1])
        input_data_1 = np.transpose(input_data_1, [0, 3, 1, 2])
        input_data_2 = np.transpose(input_data_2, [0, 3, 1, 2])
        print("input data 1 shape is: ", input_data_1.shape)
        print("input data 2 shape is: ", input_data_2.shape)
        raw_result = session.run([], {input_name_1: input_data_1, input_name_2: input_data_2})
        cls_score = np.array(raw_result[0])
        bbox_pred = np.array(raw_result[1])
        print("cls_score shape is: ", cls_score.shape)
        print("bbox_pred shape is: ", bbox_pred.shape)
        if output_txt:   
            f = open("cls_score_onnx.txt", "w")
            for i in range(0, cls_score.shape[0]):
                for j in range(0, cls_score.shape[1]):
                    for k in range(0, cls_score.shape[2]):
                        for m in range(0, cls_score.shape[3]):
                            f.write(str(cls_score[i, j, k, m])+'\n')
            f.close()

            f = open("bbox_pred_onnx.txt", "w")
            for i in range(0, bbox_pred.shape[0]):
                for j in range(0, bbox_pred.shape[1]):
                    for k in range(0, bbox_pred.shape[2]):
                        for m in range(0, bbox_pred.shape[3]):
                            f.write(str(bbox_pred[i, j, k, m])+'\n')
            f.close()





if __name__ == '__main__':

    # convert model
    model_simp, check = simplify(ONNX_MODEL)
    model_simp.ir_version = 3
    assert check, "Simplified ONNX model could not be validated"
    onnx.save_model(model_simp, ONNXSIM_MODEL)
    # Get onnx output
    print('--> Get onnx output')
    onnx_predict(ONNXSIM_MODEL, IMG_FILE, output_txt=True)

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    # input RGB, model RGB
    if MODEL==0 or MODEL==1:
        rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.3942, 57.12, 57.3756]], reorder_channel='0 1 2', target_platform=["rv1109"])
    else:
        MEAN = [0]*96
        STD = [1]*96 # 减去均值除以标准差，这里并不做改变
        rknn.config(mean_values=[MEAN, MEAN], std_values=[STD, STD], target_platform=["rv1109"]) # 两个MEAN代表两个输入
    print('done')

    # Load onnx model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNXSIM_MODEL)
    if ret != 0:
        print('Load LightTrack failed!')
        exit(ret)
    print('done')

    if do_quantization:
        # # Hybrid quantization step1
        # print('--> hybrid_quantization_step1')
        # ret = rknn.hybrid_quantization_step1(dataset=dataset)
        # if ret != 0:
        #     print('hybrid_quantization_step1 failed!')
        #     exit(ret)
        # print('done')

        # Hybrid quantization step2
        print('--> Hybrid quantization step2')
        ret = rknn.hybrid_quantization_step2(dataset=dataset, model_input="torchjitexport.json", data_input="torchjitexport.data", model_quantization_cfg="torchjitexport.quantization.cfg", pre_compile=False)
        if ret != 0:
            print('step2 failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            print('Export LightTrack.rknn failed!')
            exit(ret)
        print('done')
    else:
        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=False, dataset='./dataset_aa_init.txt', pre_compile=False)
        if ret != 0:
            print('Build LightTrack failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL_FLOAT32)
        if ret != 0:
            print('Export LightTrack.rknn failed!')
            exit(ret)
        print('done')

    if do_analysis and do_quantization:
        print('--> Accuracy analysis')
        rknn.accuracy_analysis(inputs=dataset_aa, target='rv1109')
        print('done')


    if do_inference:
        print('--> Init runtime')
        ret = rknn.init_runtime(target='rv1109', device_id="6383a316ecd249aa")
        if ret != 0:
            print(ret)
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # Set inputs
        print('--> Running model')
        if MODEL==2:
            input_data_1 = np.load(IMG_FILE.split(' ')[0])
            input_data_2 = np.load(IMG_FILE.split(' ')[1])
            inputs = [input_data_1, input_data_2]
            outputs = rknn.inference(inputs=inputs)   # 这里要格外注意，默认是“NHWC”，输入数据npy也是NHWC格式
            cls_score, bbox_pred = outputs[0], outputs[1]
            N, C, H, W = cls_score.shape
            print("cls_score shape is: ", cls_score.shape)
            f = open("cls_score_rknn.txt", "w")
            for i in range(N):
                for j in range(C):
                    for k in range(H):
                        for m in range(W):
                            f.write(str(cls_score[i, j, k, m])+"\n")
            f.close()

            N, C, H, W = bbox_pred.shape
            print("bbox_pred shape is: ", bbox_pred.shape)
            f = open("bbox_pred_rknn.txt", "w")
            for i in range(N):
                for j in range(C):
                    for k in range(H):
                        for m in range(W):
                            f.write(str(bbox_pred[i, j, k, m])+"\n")

            f.close()
        elif MODEL==0:
            input_data = cv2.imread(IMG_FILE)
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            print("rknn input data shape is: ", input_data.shape)   # 输入格式为（HWC）--> (127, 127, 3)
            inputs = [input_data]
            outputs = rknn.inference(inputs=inputs)
            zf= outputs[0]
            N, C, H, W = zf.shape
            print("zf shape is: ", zf.shape)
            f = open("zf_rknn.txt", "w")
            for i in range(N):
                for j in range(C):
                    for k in range(H):
                        for m in range(W):
                            f.write(str(zf[i, j, k, m])+"\n")
            f.close()
        elif MODEL==1:
            input_data = cv2.imread(IMG_FILE)
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            print("rknn input data shape is: ", input_data.shape)
            inptus = [input_data]
            outputs = rknn.inference(inputs=inputs)
            xf= outputs[0]
            N, C, H, W = xf.shape
            print("xf shape is: ", xf.shape)
            f = open("xf_rknn.txt", "w")
            for i in range(N):
                for j in range(C):
                    for k in range(H):
                        for m in range(W):
                            f.write(str(xf[i, j, k, m])+"\n")
            f.close()
        print("done")

        if MODEL==2:
            # onnx output and rknn output
            file1 = "cls_score_rknn.txt"
            file2 = "cls_score_onnx.txt"
            ss = consine(file1, file2)
            print("cls_score consine norm is: ", ss)

            file1 = "bbox_pred_rknn.txt"
            file2 = "bbox_pred_onnx.txt"
            ss = consine(file1, file2)
            print("bbox_pred consine norm is: ", ss)

    if do_eval_perf:
        # perf
        print('--> Begin evaluate model performance')
        perf_results = rknn.eval_perf(inputs=inputs)
        print('done')

    rknn.release()