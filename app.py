import torch
import os
import subprocess
import json
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from videoClassification.model import generate_model
from videoClassification.classify import classify_video
from configuration import Configuration
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
publicDirectory = os.path.join(BASE_DIR, "public")
if not os.path.exists(publicDirectory):
   os.makedirs(publicDirectory)

app = Flask(__name__,static_url_path="")
config = Configuration()
model = None


model = generate_model(config)
print('loading model {}'.format(config.model))
config.model = './resnext-101-kinetics.pth'
model_data = torch.load(config.model)
assert config.arch == model_data['arch']
model.load_state_dict(model_data['state_dict'])
model.eval()
if config.verbose:
    print(model)
ffmpeg_loglevel = 'quiet'
if config.verbose:
    ffmpeg_loglevel = 'info'

def transformDf(row):
    data_frame_dict = dict()
    video_path = row.video
    list_segment = [tuple(info["segment"]) for info in row.clips]
    data_frame_dict["video"] = video_path
    data_frame_dict["frame_inicio"] = list(map(lambda x : x[0] , list_segment))
    data_frame_dict["frame_fin"] = list(map(lambda x : x[1] , list_segment))
    data_frame_dict["descriptor"] = [info["features"] for info in row.clips]
    return pd.DataFrame.from_dict(data_frame_dict)

def clustering_(dfFinal):
    descriptor_array = dfFinal.descriptor.apply(lambda x: np.array(x)).to_numpy()
    descriptor_array = np.vstack(descriptor_array).astype(np.float32)
    clustering = KMeans(n_clusters=3, n_jobs=-1).fit(descriptor_array)
    dfFinal['labels'] = clustering.predict(descriptor_array)
    return descriptor_array, clustering

def get_min_seg(df, cluste):
    tuple_min_pos = list()
    for i in range(3):
        d = np.vstack(df.loc[df.labels == i].descriptor.apply(lambda x: np.array(x)).to_numpy())
        f = cluste.cluster_centers_[i]
        g = (d-f)**2
        valorminimo = np.argmin(np.sum(g,axis=1))
        tuple_min_pos.append(df.loc[df.labels == i].iloc[valorminimo])
    return tuple_min_pos

def get_name_frame_files(array_segmen):
    file_fase_dict = dict()
    file_fase_dict['fase-0'] = ''
    file_fase_dict['fase-1'] = ''
    file_fase_dict['fase-2'] = ''
    list_frames = list()
    for i, value in enumerate(array_segmen):
        frame = (value.frame_inicio + value.frame_fin)/2
        list_frames.append(frame)

    list_frames.sort()

    for i, _value in enumerate(list_frames):
        file_fase_dict['fase-%d'%i] = 'image_%05d.jpg' % _value

    return file_fase_dict

@app.route('/getImage', methods=['POST'])
def getImage():
    if request.method == 'POST':
        file_name = request.form['filename']
        return send_file('tmp/%s' % file_name)


@app.route('/splitVideo',methods=['POST'])
def splitVideo():
    response = dict()
    if request.method == 'POST':
        file = request.files['video']
        if file.filename == '':
            response['message'] = "Error"
            return jsonify(response)
        else :

            process_video_folder = config.save_folder

            if os.path.exists(process_video_folder):
                subprocess.call("rm -rf %s" % process_video_folder , shell=True)
            os.mkdir(process_video_folder)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
            basepath, 'static', secure_filename(file.filename))
            file.save(file_path)
            if os.path.exists('tmp'):
                subprocess.call('rm -rf tmp', shell=True)
            subprocess.call('mkdir tmp', shell=True)

            subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(file_path),
                            shell=True)
            result = classify_video('tmp', file.filename, '', model, config)
            with open('resultado.json','w') as outfile:
                json.dump(result,outfile)

            a = pd.read_json("resultado.json")
            dfFinal = transformDf(a)
            descriptor_array,clustering = clustering_(dfFinal)

            array_segmen = get_min_seg(dfFinal,clustering)

            return jsonify(get_name_frame_files(array_segmen))

if __name__ == '__main__':
    app.run(port=6008)
