from videoClassification.mean import get_mean
import multiprocessing 

class Configuration:
    def __init__(self):
        self.input = 'input'
        self.video_root = ''
        self.model = ''
        self.output = 'output.json'
        self.mode = 'feature'
        self.batch_size = 32
        self.n_threads = 4
        self.model_name = 'resnext'
        self.model_depth = 101
        self.resnet_shortcut = 'B'
        self.wide_resnet_k = 2
        self.resnext_cardinality = 32
        self.no_cuda = False
        self.verbose = False
        self.mean = get_mean()
        self.arch = '{}-{}'.format(self.model_name, self.model_depth)
        self.sample_size = 144
        self.sample_duration = 16
        self.n_classes = 400
        self.overlapping = 5
        self.sample_duration = 11
        self.save_folder = 'videoPro'
