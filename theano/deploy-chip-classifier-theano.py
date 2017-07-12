import logging
import ast, os, time
import json
import numpy as np
import cv2
import subprocess

from keras.models import load_model
from shutil import copy, copyfile
from gbdx_task_interface import GbdxTaskInterface
from os.path import join

def resnet_preproc_keras(data_r, intensities):
    """
    Args: data_r: list of images
          intensities: list of R,G,B intensities to subtract from RGB bands
    Returns: data
          data: list of mean-adjusted & RGB -> BGR transformed images
    """
    # RGB -> BGR:
    data = data_r[:, :, :, [2,1,0]]
    data[:, :, :, 0] -= intensities[2]
    data[:, :, :, 1] -= intensities[1]
    data[:, :, :, 2] -= intensities[0]

    # TF -> TH dim ordering
    data = data.swapaxes(1,-1)

    return data

def resize_image(path, rows, cols):
    """
    Args: path [string], rows, cols [int]
        path: path of an image; rows, cols - columns and rows
        of resized image
    Returns: resized [numpy array rows x columns x 3]
        resized image
	(if the image is invalid, it returns a rowsxcols array of zeros)
    """
    img = cv2.imread(path)
    try:
        resized = cv2.resize(img, (rows, cols))
    except:
        print 'Resizing can not be performed. Corrupt chip?'
        resized = np.zeros([rows, cols, 3], dtype=int)
    return resized


class DeployClassifier(GbdxTaskInterface):

    def __init__(self):
        '''
        Retrieve task inputs and unzip image chips.
        '''
        init_start = time.time()
        GbdxTaskInterface.__init__(self)

        # Make output directory and define output filename
        self.outdir = self.get_output_data_port('results')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.out_file = join(self.outdir, 'classified.json')

        # Make logs directory
        self.logsdir = self.get_output_data_port('logs')
        if not os.path.exists(self.logsdir):
            os.makedirs(self.logsdir)
        logging.basicConfig(filename=join(self.logsdir, 'out.log'), level=logging.DEBUG)

        # Get string inputs
        self.classes = self.get_input_string_port('classes', default=None)
        self.size = int(self.get_input_string_port('size', default='224'))
        self.deploy_batch = int(self.get_input_string_port('deploy_batch', default='100'))
        self.normalization_vector = map(float, self.get_input_string_port('normalization_vector', default='123.68,116.779,103.939').split(','))

        # Get input directories
        self.chip_dir = self.get_input_data_port('chips')
        self.model_dir = self.get_input_data_port('model')

        # Unzip chips
        print 'Unzipping chips'
        logging.debug('Unzipping chips')
        archive = [f for f in os.listdir(self.chip_dir) if f.endswith('.tar')][0]
        command = 'tar xvf ' + join(self.chip_dir, archive) + ' -C ' + self.chip_dir
        proc = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        self.chip_dir = join(self.chip_dir, archive[:-4])    # update chip directory

        # Get files in input directories
        self.chips = [img for img in os.listdir(self.chip_dir) if
                     (img.endswith('.jpg') or img.endswith('.tif') or img.endswith('.png'))]
        self.model = [mod for mod in os.listdir(self.model_dir) if mod.endswith('.h5')][0]

        # Move model under chip_dir and make it the working directory
        copyfile(join(self.model_dir, self.model), join(self.chip_dir, 'model.h5'))
        os.chdir(self.chip_dir)
        self.model = 'model.h5'

        init_end = time.time()
        print 'Total initialization time: {}s'.format(str(init_end - init_start))
        logging.debug('Total initialization time: {}s'.format(str(init_end - init_start)))

    def deploy_model(self, model):
        '''
        Deploy model.
        '''
        yprob, classed_json = [], {}

        # Format classes: ['class_1', 'class_2']
        if self.classes:
            self.classes = [clss.strip() for clss in self.classes.split(',')]
        else:
            self.classes = [str(num) for num in xrange(model.output_shape[-1])]

        # Classify chips in batches
        indices = np.arange(0, len(self.chips), self.deploy_batch)
        no_batches = len(indices)

        for no, index in enumerate(indices):
            this_batch = self.chips[index: (index + self.deploy_batch)]

            # Resize and preprocess
            chips = resnet_preproc_keras(np.array([resize_image(chip_name, self.size, self.size) for chip_name in this_batch],
                                         dtype='float32'), self.normalization_vector)

            print 'Classifying batch {} of {}'.format(no+1, no_batches)
            t1 = time.time()
            yprob += list(model.predict_on_batch(chips))
            t2 = time.time()
            print 'Batch classification time: {}s'.format(t2-t1)
            logging.debug('Batch classification time: {}s'.format(t2-t1))

        # Get predicted classes and certainty
        yhat = [self.classes[np.argmax(i)] for i in yprob]
        ycert = [round(float(np.max(j)), 10) for j in yprob]

        # Create json with classes for each chip
        data = zip(yhat, ycert)
        for ix, chip in enumerate(self.chips):
            classed_json[chip[:-4]] = {'class': yhat[ix],
                                       'certainty': ycert[ix]}

        # Save classed json to output file location
        with open(self.out_file, 'wb') as f:
            json.dump(classed_json, f)


    def invoke(self):
        '''
        Execute task
        '''

        # Load model
        load_start = time.time()
        print 'Loading model...'
        logging.debug('Loading model...')
        model = load_model(self.model)
        load_end = time.time()
        print 'Took {} seconds to load model'.format(str(load_end - load_start))
        logging.debug('Took {} seconds to load model'.format(str(load_end-load_start)))

        # Deploy model
        print 'Deploying model...'
        logging.debug('Deploying model...')
        deploy_start = time.time()
        self.deploy_model(model)
        deploy_end = time.time()
        print 'Total classification time: {}'.format(deploy_end - deploy_start)
        logging.debug('Total classification time: {}'.format(deploy_end - deploy_start))

if __name__ == '__main__':
    with DeployClassifier() as task:
        task.invoke()
