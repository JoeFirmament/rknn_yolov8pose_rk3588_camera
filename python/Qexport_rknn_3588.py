from rknn.api import RKNN



if __name__ == '__main__':
    rknn = RKNN(verbose=True,verbose_file='../model/export_rknn_3588.log')

    rknn.release()

