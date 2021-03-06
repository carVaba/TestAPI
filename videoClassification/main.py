import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from .opts import parse_opts
from .model import generate_model
from .mean import get_mean
from .classify import classify_video
#import cv2

import os


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    #Usage -> https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console?page=1&tab=votes#tab-top
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__=="__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #LOAD HOG
    ##hog = cv2.HOGDescriptor()
    ##hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hog = None
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 144
    opt.n_classes = 400
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    torch.cuda.set_device(0)
    print(torch.cuda.current_device())

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('./../class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    outputs = []

    process_video_folder = opt.save_folder

    if os.path.exists(process_video_folder):
        subprocess.call("rm -rf %s" % process_video_folder , shell=True)
    os.mkdir(process_video_folder)
    l , i = len(input_files) , 0
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=100)
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            subprocess.call("mkdir %s/%s" % (process_video_folder,input_file), shell=True)
            subprocess.call(['ffmpeg' , '-i' , video_path , '-loglevel' , 'quiet' ,process_video_folder+'/'+input_file+'/image_%05d.jpg'])
            try:
                input_file = "%s/%s" % (process_video_folder , input_file)
                result = classify_video(input_file, input_file, class_names, model, opt , hog)

                outputs.append(result)
            except Exception as ex:
            #print("error")
                print(ex)

        else:
            print("%s does not exist" % input_file)

        printProgressBar(i + 1, l, prefix='Progress:',suffix='Complete', length=100)
        i += 1

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
