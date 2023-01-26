import cv2
from rknn.api import RKNN


MODEL = 2
quantization_MODEL = True

if MODEL == 0:
    RKNN_MODEL = 'model/lighttrack_init.rknn'
    RKNN_MODEL_FLOAT32 = 'model/lighttrack_init_f32.rknn'
    RKNN_PRECOMPILE_MODEL = 'model/lighttrack_init.precompile.rknn'
    CRYPT_MODEL = 'model/lighttrack_init.crypt.rknn'

if MODEL == 1:
    RKNN_MODEL = 'model/lighttrack_backbone.rknn'
    RKNN_MODEL_FLOAT32 = 'model/lighttrack_backbone_f32.rknn'
    RKNN_PRECOMPILE_MODEL = 'model/lighttrack_backbone.precompile.rknn'
    CRYPT_MODEL = 'model/lighttrack_backbone.crypt.rknn'

if MODEL == 2:
    RKNN_MODEL = 'model/lighttrack_neck_head.rknn'
    RKNN_MODEL_FLOAT32 = 'model/lighttrack_neck_head_f32.rknn'
    RKNN_PRECOMPILE_MODEL = 'model/lighttrack_neck_head.precompile.rknn'
    CRYPT_MODEL = 'model/lighttrack_neck_head.crypt.rknn'


if __name__ == '__main__':
    rknn_pre = RKNN(verbose=True)

    if quantization_MODEL:
        # load rknn model
        ret = rknn_pre.load_rknn(RKNN_MODEL)
        if ret != 0:
            print('Load RKNN model failed.')
            exit(ret)
    else:
        # load rknn model
        ret = rknn_pre.load_rknn(RKNN_MODEL_FLOAT32)
        if ret != 0:
            print('Load RKNN model failed.')
            exit(ret)
    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn_pre.init_runtime(target='rv1109', rknn2precompile=True, device_id='6383a316ecd249aa')
    if ret != 0:
        print(ret)
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export Precompile RKNN model')
    ret = rknn_pre.export_rknn_precompile_model(RKNN_PRECOMPILE_MODEL)
    if ret != 0:
        print('Export scrfd.rknn failed!')
        exit(ret)
    print('done')

    rknn_pre.release()
