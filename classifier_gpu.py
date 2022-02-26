""" Classifies MMS FPI/DIS observations using a pre-trained CNN.
Computes PCA over the labelled data.

Part of the machine learning project concerning the observations by FPI/DIS and FPI/DES.

(c) Vyacheslav Olshevsky (2019)

"""

import pycuda.autoprimaryctx
import pycuda.gpuarray as gpuarray
from skcuda.linalg import PCA as cuPCA
import numpy as np
from matplotlib import pyplot as plt
import mms_utils as mu
import mms_plots as mplt
import os, cdflib, pickle
from glob import glob
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
import time
import  pycuda.autoprimaryctx
# imports for ConvNet
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend

pycuda.tools.clear_context_caches()
# Path to the database with MMS data
data_path = r'C:\\Users\\Davis\\data\\' 
# When working on IRFU's Brain computer, base_path is in my localhome
base_path = r'C:\\Users\\Davis\\Desktop\\seniorresearch\\volshevsky-mmslearning-davisWIP\\' 

# Date of interest
year = '2017'
month = '12'
date = year + month
print(date)
# The index of the spacecraft, 1..4
mms_index = 1
# String to prepend to filenames
mms_name = r'mms' + str(mms_index)
# Spectrograph(m?) name 'dis' - ions, 'des' - electrons.
spc_name = r'dis'
# The name of the variable in the CDF file which contains the distribution function.
var_name = mms_name + '_' + spc_name + '_dist' + '_fast'

# Path to 1 month of FPI data
fpi_data_path = os.path.join(data_path, r'mms\\' + mms_name + r'\\fpi\\fast\\l2\\' + spc_name + r'-dist\\' + year + r'\\' + month)
# FGM data
fgm_data_path = os.path.join(data_path, r'mms\\' + mms_name + r'\\fgm\\srvy\\l2\\' + year + r'\\' + month)
# Where the resulting labels are stored
model_date = '201712'
lbl_path = os.path.join(base_path, r'labels_human\\' + r'\\')
# Where the trained models are stored
model_path = os.path.join(base_path, 'models')
# Choose the model used for classification/compressing: distnet_dis_201711, autoenc3d_dis_201711
model_filename = os.path.join(model_path, r'cnn_dis_' + model_date + '_verify.h5')

# Plotting parameters
params = {'axes.labelsize': 'medium',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'font.size': 25,
          'font.family': 'sans-serif',
          'text.usetex': False,
          'mathtext.fontset': 'stixsans',}
plt.rcParams.update(params)

def classify_one_file(fpi_filepath, model, lbl_cdf_file):
    """ Returns labels for a single CDF file.

    """
    # Read energy distribution fom CDF file
    fpi_cdf_file = cdflib.cdfread.CDF(fpi_filepath)
    dist = fpi_cdf_file.varget(var_name)
    # TODO: for an unknown yet reason, the order of dimensions on Windows and on Brain is different, therefore
    # for a model, trained on Windows, we do swapaxes on Brain
    dist = dist.swapaxes(1, 3)
    epoch = fpi_cdf_file.varget('Epoch')
    fpi_cdf_file.close()

    # Prepare data for classificator: normalize and add extra dimension - 'channel'
    dist = mu.normalize_data(dist, verbose=False)
    dist = dist.reshape(dist.shape + (1,))

    # Probability for each class
    predictions = model.predict(dist)
    # The most probable class
    label = predictions.argmax(axis=1)

    mu.add_labels_CDF(epoch, label, predictions, fpi_filepath, lbl_cdf_file)

def classify_one_month(fpi_path, lbl_path, model_file):
    """ CDF files with FPI data contain 2 hours of data or less.
    We will write 1 CDF file with labels per month (1 folder with CDF files in MMS catalogue).

    Example:

    classify_one_month(fpi_data_path, lbl_path, model_filename)

    """
    backend.set_image_data_format('channels_last')

    if not(os.path.exists(fpi_path)):
        raise FileExistsError('The specified fpi_path does not exist!\n'+fpi_path)

    # Load Keras model from HDF file saved by Keras.
    model = load_model(model_file)

    # Search for all files in that folder
    fpi_names = glob(os.path.join(fpi_path, mms_name + r'*fpi_fast_l2_' + spc_name + r'-dist_*.cdf'))
    fpi_names.sort()
    if len(fpi_names) < 1:
        raise FileNotFoundError('No matching files in fpi_path!')

    # Read first file to get its header
    fpi_cdf_file = cdflib.cdfread.CDF(fpi_names[0])
    fpi_spec = fpi_cdf_file.cdf_info()
    fpi_cdf_file.close()

    # Create a CDF file with labels
    lbl_spec = {'Copyright': 'Vyacheslav Olshevsky (slavik@kth.se)'}
    for k in ['Majority', 'Encoding', 'Checksum', 'rDim_sizes', 'Compressed']:
        lbl_spec[k] = fpi_spec[k]

    os.makedirs(lbl_path, exist_ok=True)

    d = mu.parse_mms_filename(fpi_names[0])
    lbl_filename = r'_'.join(['labels', d['instrument'], d['tmMode'], d['detector'], d['product'], d['timestamp'][:6]]) + r'.cdf'

    # Create an empty CDF file for labels
    lbl_cdf_file = cdflib.cdfwrite.CDF(os.path.join(lbl_path, lbl_filename), cdf_spec=lbl_spec, delete=True)

    # Write new label variables for each CDF file found in data
    for f in fpi_names:
        classify_one_file(f, model, lbl_cdf_file)

    lbl_cdf_file.close()
    print('Written file', lbl_cdf_file.path.as_posix())
    return lbl_cdf_file

def autoencode_one_file(fpi_filepath, model, normalizer, encoder):
    """ Returns labels for a single CDF file.

    Keywords:
        compressed_layer - the index of the compressed layer in the encoder-decoder model

    """
    # Read energy distribution fom CDF file
    print('Reading', fpi_filepath)
    fpi_cdf_file = cdflib.cdfread.CDF(fpi_filepath)
    dist = fpi_cdf_file.varget(var_name)
    epoch = fpi_cdf_file.varget('Epoch')
    fpi_cdf_file.close()

    # Prepare data for passing through CNN
    dist = normalizer(dist)

    # Compressed representation, should be [#epochs, 128]
    compressed = encoder([dist])[0]

    # Now, pass data through the whole CNN to compute the difference input-output
    predictions = model.predict(dist)
    diff = ((dist - predictions)**2).sum(axis=tuple(range(1, dist.ndim)))
    diff = (diff / (dist**2).sum(axis=tuple(range(1, dist.ndim))))**0.5

    return compressed, diff, epoch

def autoencode_one_month(fpi_path, model_file, compressed_layer=6, n_clusters=len(mu.regions)-1):
    """ CDF files with FPI data contain 2 hours of data or less.
    We will write 1 CDF file with labels per month (1 folder with CDF files in MMS catalogue).

    Example:

    aut_dict = autoencode_one_month(fpi_data_path, model_filename)

    """
    if not(os.path.exists(fpi_path)):
        raise FileExistsError('The specified fpi_path does not exist!\n'+fpi_path)

    # imports for autoencoder
    from tensorflow import keras
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras import backend
    backend.set_image_data_format('channels_first')

    # Normalization is different from ConvNet
    from auto_classifier import normalize_data

    # Load Keras model from HDF file saved by Keras.
    model = load_model(model_file)
    # The function that returns output of the bottleneck layer
    encoder = keras.backend.function([model.layers[0].input], [model.layers[compressed_layer].output])

    # Search for all files in that folder
    fpi_names = glob(os.path.join(fpi_path, mms_name + r'*fpi_fast_l2_' + spc_name + r'-dist_*.cdf'))
    fpi_names.sort()
    if len(fpi_names) < 1:
        raise FileNotFoundError('No matching files in fpi_path!')

    # Read first file to get its header for the new CDF file
    fpi_cdf_file = cdflib.cdfread.CDF(fpi_names[0])
    fpi_spec = fpi_cdf_file.cdf_info()
    fpi_cdf_file.close()

    # Pass files through encoder one-by-one. TODO: remove hardcoded compressed-layer dimension.
    compressed, difference, epoch = np.empty((0,128), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    for f in fpi_names:
        c, d, e = autoencode_one_file(f, model, normalize_data, encoder)
        compressed = np.concatenate([compressed, c])
        difference = np.concatenate([difference, d])
        epoch = np.concatenate([epoch, e])

    # Compute k-Means clustering of compressed representation
    print('Computing K-means')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster = kmeans.fit_predict(compressed).astype(np.int8)

    # Create an empty CDF file for label
    d = mu.parse_mms_filename(fpi_names[0])
    lbl_filename = r'_'.join(['autoencoded', d['instrument'], d['tmMode'], d['detector'], d['product'], d['timestamp'][:6]]) + r'.cdf'

    os.makedirs(lbl_path, exist_ok=True)
    lbl_filename = os.path.join(lbl_path, lbl_filename)

    # Create a CDF file with labels
    lbl_spec = {'Copyright': 'Vyacheslav Olshevsky (slavik@kth.se)'}
    for k in ['Majority', 'Encoding', 'Checksum', 'rDim_sizes', 'Compressed']:
        lbl_spec[k] = fpi_spec[k]

    aut_dict = {'compressed': compressed, 'difference': difference, 'cluster': cluster, 'epoch': epoch, 'id': mu.epoch2id(epoch, mms_index)}
    mu.write_dict_cdf(lbl_filename, aut_dict)

    return aut_dict

def demo_labels(lbl_path, date, lbl_dict=None, reg_dict=None):
    """ Plots with labels, IRFU regions, mislabelled data, etc.

    Example:
        lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + date + '.cdf'))
        reg_dict = mu.read_irfu_regions(r'C:/Projects/MachineLearningSpace/data/mms/irfu/cal/mms1/edp/sdp/regions/mms1_edp_sdp_regions_20171206_v0.0.0.txt')
        demo_labels(lbl_path, date, lbl_dict=lbl_dict, reg_dict=reg_dict)

    """
    if lbl_dict is None:
        lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + date + '.cdf'))

    # Scatter-plot of region labels
    plt.ion()
    fig = plt.figure('Labelled data', figsize=(20, 15),)
    ax = fig.add_subplot(1, 1, 1)

    lbl_time = mu.epoch2time(lbl_dict['epoch'])
    for c in sorted(list(mu.regions.keys())):
        i = np.where(lbl_dict['label'] == c)
        if len(i[0]) > 0:
            ax.scatter(lbl_time[i], lbl_dict['label'][i], s=100, marker='o', c=mu.region_colors[c])

    ax.tick_params(axis='x', rotation=30)

    # Overplot IRFU regions
    if not (reg_dict is None):
        xlim = ax.get_xlim()

        reg_epoch = reg_dict['epoch']
        reg_label = reg_dict['region']
        # Note, general Msheath translates into q-par!
        reg_label = np.vectorize(mu.irfu_to_my_regions.get)(reg_label)

        plt_epoch = np.empty(reg_epoch.shape[0]*2 - 1)
        plt_epoch[::2] = reg_epoch
        plt_epoch[1::2] = reg_epoch[1:]

        plt_label = np.empty(plt_epoch.shape[0])
        plt_label[::2] = reg_label
        plt_label[1::2] = reg_label[:-1]

        plt_time = mu.epoch2time(plt_epoch)
        ax.plot(plt_time, plt_label, color='grey', linewidth=2, label='IRFU labelled')
        ax.set_xlim(xlim)

    yticks = np.array(list(mu.regions.keys()))
    ax.set_ylim(yticks.min()-0.5, yticks.max()+0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([mu.regions[k] for k in mu.regions.keys()])
    ax.set_ylim(-0.2, 4.2)

    plt.tight_layout()
    plt.show()

def labels_accuracy(lbl_dict, reg_dict):
    """ Compare accuracy of labels and IRFU regions.
    We map our 5 classes of regions onto marked 3 classes.

    Example:
        lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + date + '.cdf'))
        reg_dict = mu.read_irfu_regions(r'C:/Projects/MachineLearningSpace/data/mms/irfu/cal/mms1/edp/sdp/regions/mms1_edp_sdp_regions_20171206_v0.0.0.txt')
        labels_accuracy(lbl_dict, reg_dict)

    """
    epoch = lbl_dict['epoch']
    label = lbl_dict['label']

    # Get regions for labelled data
    my_region = np.vectorize(mu.my_to_irfu_regions.get)(label)

    # Map labelled periods to each epoch
    region = mu.map_interval_labels(reg_dict['region'], reg_dict['epoch'], epoch)

    print('Total number of examples', epoch.shape[0])
    for k, v in mu.irfu_regions.items():
        mine = np.where(my_region == k)
        irfu = np.where(region == k)
        mislabld = np.where(my_region[irfu] != k)
        falsepred = np.where(region[mine] != k)
        print(k, v + '.', 'Labelled by me %i. Labelled by IRFU %i' % (mine[0].shape[0], irfu[0].shape[0]))
        print('Error: %4.2F' % (mislabld[0].shape[0] / irfu[0].shape[0]))
        print('False predictions: %4.2F\n' % (falsepred[0].shape[0] / mine[0].shape[0]))

def PCA_labelled(fpi_path, labels_dict, n_components=2, pic_filename=''):
    """ Compute a PCA over a large amount of labelled data.

    Example:
        lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + year + month + '.cdf'))
        pca_dict = PCA_labelled(fpi_data_path, lbl_dict, v name=os.path.join(base_path, r'pca_' + year + month + '.pic'))

    """
    

    labels_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + '201711' + '.cdf'))
    fpi_dict = mu.read_many_fpi_cdf(fpi_path, vars=['epoch', var_name])

    # Sanity check for loaded data

#    if np.abs((fpi_dict['epoch'] - labels_dict['epoch'])).sum() > 0:
#        raise ValueError('Epochs from labelled data and from FPI data are in different order!')

    pca_data = fpi_dict[var_name]
    pca_data = mu.normalize_data(fpi_dict[var_name])
    pca_data = pca_data.reshape((pca_data.shape[0], np.prod(pca_data.shape[1:])))
    print(pca_data.dtype)
    X_gpu = gpuarray.GPUArray(pca_data.shape, dtype=np.float32, order='F')
    X_gpu.set(pca_data)
    cupca = cuPCA(n_components=n_components)   
    projection = cupca.fit_transform(X_gpu)
    T_gpu = cupca.fit_transform(X_gpu)
    projection = np.empty(T_gpu.shape, dtype=np.float32)

    T_gpu.get(projection, pagelocked=False)  
    pca_dict = {'projection': projection, 'label': labels_dict['label'], 'epoch': labels_dict['epoch']} #, 'pca': pca}
 #   for i in range(n_components):
#        print('Variance explained by the %i component:' % i, pca.explained_variance_[i]/sum(pca.explained_variance_))


    if len(pic_filename) > 0:
        with open(pic_filename, 'wb') as pic_file:
            print('Pickling labelled PCA in', pic_file)
            pickle.dump(pca_dict, pic_file)

    return pca_dict

def plot_rb_correlation(fgm_dict, lbl_dict, plot_labels=range(-1, len(mu.regions)-1)):
    """ Some nice 2D colorful plot to demonstrate how powerful labelling is.

    I have only played with this plot once, it is rather immature.

    """
    # Demo with FGM data plot
    epoch_lbl = lbl_dict['epoch']
    epoch_fgm = fgm_dict['epoch']
    epoch_state_fgm = fgm_dict['epoch_state']

    for k in ['mms1_fgm_b_gsm_srvy_l2', 'mms1_fgm_r_gse_srvy_l2']:
        v = fgm_dict[k]
        interpolator = interp1d(epoch_fgm, v, axis=0) if v.shape[0] == epoch_fgm.shape[0] else interp1d(epoch_state_fgm, v, axis=0)
        fgm_dict[k] = interpolator(epoch_lbl)

    r = fgm_dict[mms_name+'_fgm_r_gse_srvy_l2'][:,3] / mu.earth_radius
    b = fgm_dict[mms_name+'_fgm_b_gsm_srvy_l2'][:,2]
    l = lbl_dict['label']

    # Plot some fun distributions
    fig = plt.figure('Correlations', figsize=(38, 19))
    ax = fig.add_subplot(1, 1, 1)
    for c in plot_labels:
        i = np.where(l == c)
        ax.scatter(r[i], b[i], s=30, marker='o', c=mu.region_colors[c])

    # Add fake points of larger size and legend
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0, y0 = xlim[0] - 1000, ylim[0] - 1000
    for c in plot_labels:
        ax.scatter([x0], [y0], c=mu.region_colors[c], s=800, label=mu.regions[c])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$r_{GSE}$ (R$_E$)')
    ax.set_ylabel(r'$B_z$ GSM [nT]')
    plt.legend(loc=0, frameon=False, borderaxespad=0.01, handletextpad=0.2)

    plt.tight_layout()
    plt.show()

def plot_probability_hist(lbl_dict):
    """ TODO: should this method go to some other file, e.g., mms_utils?

    """
    fig = plt.figure('Class probability histogram', figsize=(38, 19))
    ax = fig.add_subplot(1, 1, 1)
    histo = ax.hist(lbl_dict['probability'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel('Probability')
    ax.set_ylabel('# of examples')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

def plot_labels_orbit(fgm_dict, lbl_dict):
    """ Plots orbits and labels along them.

    Example:
        lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + date + '.cdf'))
        fgm_dict = mu.read_many_fgm_cdf(fgm_data_path)
        plot_labels_orbit(fgm_dict, lbl_dict)

    """
    # Plot the orbits
    epoch_lbl = lbl_dict['epoch']
    prob = lbl_dict['probability'].max(axis=1)
    labels = lbl_dict['label']
    colors = np.array([mu.region_colors[l] for l in labels])
    r = fgm_dict['mms1_fgm_r_gse_srvy_l2'] / mu.earth_radius
    # The epoch for radius measurements
    epoch_state = fgm_dict['epoch_state']
    # Interpolate coordinates on epoch_lbl
    X = interp1d(epoch_state, r[:,0], axis=0)(epoch_lbl)
    Y = interp1d(epoch_state, r[:,1], axis=0)(epoch_lbl)

    fig = plt.figure('Regions on orbit', figsize=(38, 19))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(r[:,0], r[:,1], color='dimgray', linewidth=2) #, label='SC trajectory')
    ax.scatter(X, Y, s=200*prob, marker='o', c=colors) #, label='Prediction')

    # Mark unknown regions
    unk = np.where(prob < 0.5)
    ax.scatter(X[unk], Y[unk], s=200, marker='o', color='yellow') #edgecolors=mu.region_colors[-1], facecolors='none')

    # Add fake points of larger size and legend
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0, y0 = xlim[0] - 1000, ylim[0] - 1000
    my_colors = dict(mu.region_colors)
    my_colors[-1] = 'yellow'
    for k, v in my_colors.items():
        ax.scatter([x0], [y0], c=v, s=800, label=mu.regions[k])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend(loc=0, frameon=False, borderaxespad=0.01, handletextpad=0.2)

    '''
    # Legend for labels
    x0, y0, dx, dy = 16, 4.8, 0.2, 1.5
    if date == '201712':
        x0, y0, dx, dy = -2, 16, 0.2, 1.5

    for i in range(-1, len(mu.regions)-1):
        clr = mu.region_colors[i] if (i > -1) else 'yellow'
        ax.scatter(x0, y0 - i*dy + 0.4*dy, s=200, marker='o', color=clr)
        ax.text(x0 + dx, y0 - i*dy, mu.regions[i], {'ha': 'left', 'va': 'bottom'},) # fontsize='small')

    '''

    ax.set_xlabel('X GSE (Re)')
    ax.set_ylabel('Y GSE (Re)')
    ax.grid()

    plt.tight_layout()
    plt.show()

def plot_PCA(projection, label, scale_x=1, scale_y=1, plot_labels=range(-1, len(mu.regions)-1), sizes=None):
    """ Scatter plots in 2 first PCA components.

    Parameters:
        projection - PCA projections
        label - labels for each element

    Keywords:
        set scale_x and scale_y to -1 to flip PCA axes
        plot_labels=[0, 3, 2, 1, -1] - which labels and in what order to plot

    Example:

    # Load the labels to color the dots on PCA diagram
    lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + date + '.cdf'))

    # Load pre-computed PCA
    with open(os.path.join(base_path, 'pca_'+date+'.pic'), 'rb') as f:
        pca_dict = pickle.load(f)

    assert(np.all(pca_dict['epoch'] == lbl_dict['epoch']))
    plot_PCA(pca_dict['projection'], lbl_dict['label'])

    """
    plt.ion()

    if sizes is None:
        sizes = [40,] * len(plot_labels)

    fig = plt.figure('PCA', figsize=(38, 19))
    ax = fig.add_subplot(1, 1, 1)

    # Plot all examples with points with given sizes
    for i in range(len(plot_labels)):
        k = plot_labels[i]
        v = mu.regions[k]
        this_label = np.where(label == k)
        if len(this_label[0]) > 0:
            ax.scatter(scale_x*projection[this_label, 0], scale_y*projection[this_label, 1], c=mu.region_colors[k], s=sizes[i])

    # Add fake points of larger size and legend
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0, y0 = xlim[0] - 1000, ylim[0] - 1000
    for i in range(len(plot_labels)):
        k = plot_labels[i]
        v = mu.regions[k]
        ax.scatter([x0], [y0], c=mu.region_colors[k], s=800, label=v)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('PCA-one')
    ax.set_ylabel('PCA-two')
    plt.legend(loc=0, frameon=False, borderaxespad=0.01, handletextpad=0.2)
    plt.tight_layout()
    plt.show()

    return ax

def label_accuracy(lbl_dict, ref_dict):
    """ Check how many mislabelled and how many false predictions in the lbl_dict compared to ref_dict.

    Examples:

    >>>
    #ref_dict = mu.convert_labels(mu.read_labels_cdf)(r'C:/Projects/MachineLearningSpace/labelled_data/model_ML_paper/labels-human/labels_fpi_fast_dis_dist_' + date + '.cdf')

    >>>
    date = '201712'
    ref_dict = mu.read_labels_cdf(r'C:/Projects/MachineLearningSpace/labelled_data/labels_fpi_fast_dis_dist_' + date + '.cdf')
    lbl_dict = mu.read_labels_cdf(r'C:/Projects/MachineLearningSpace/data/labels_201712_verify01/labels_fpi_fast_dis_dist_' + date + '.cdf')
    label_accuracy(lbl_dict, ref_dict)

    """
    assert(np.all(lbl_dict['epoch'] == ref_dict['epoch']))

    # First, choose only the 'known' examples
    lbl = lbl_dict['label']
    ref_lbl = ref_dict['label']

    known = np.where(ref_lbl > -1)
    lbl = lbl[known]
    ref_lbl = ref_lbl[known]

    print('Total number of known examples', lbl.shape[0])
    print('Overall error %4.2F%%' % (100. * np.where(lbl != ref_lbl)[0].shape[0] / lbl.shape[0]))
    total_mis, total_false = 0, 0
    for k, v in mu.regions.items():
        predicted = np.where(lbl == k)
        human_labelled = np.where(ref_lbl == k)
        mislabelled = np.where(lbl[human_labelled] != k)
        falsepred = np.where(ref_lbl[predicted] != k)

        predicted, human_labelled, mislabelled, falsepred = predicted[0].shape[0], human_labelled[0].shape[0], mislabelled[0].shape[0], falsepred[0].shape[0]
        total_mis += mislabelled
        total_false += falsepred

        print(k, v + '.', '\nHuman-labelled %i; Predicted %i; Mislabelled %i; False predictions %i' % (human_labelled, predicted, mislabelled, falsepred))
        if human_labelled > 0:
            print('Error: %4.2F%%' % (100. * mislabelled / human_labelled))
        if predicted > 0:
            print('False predictions: %4.2F%%\n' % (100.*falsepred / predicted))

    print('Total: mislabelled %i; false %i' % (total_mis, total_false))
    print('Percentage mislabelled %4.2F%%; percentage false %4.2F%%' % (100.*total_mis / lbl.shape[0], 100.*total_false / lbl.shape[0]))

def label_confusion(lbl_dict, ref_dict):
    """ Check how many mislabelled and how many false predictions in the lbl_dict compared to ref_dict.

    Examples:

    >>>
    ref_dict = mu.convert_labels(mu.read_labels_cdf)(r'C:/Projects/MachineLearningSpace/labelled_data/model_ML_paper/labels-human/labels_fpi_fast_dis_dist_' + date + '.cdf')
    lbl_dict = mu.read_labels_cdf(r'C:/Projects/MachineLearningSpace/labelled_data/model_ML_paper/labels-CNN/labels_fpi_fast_dis_dist_' + date + '.cdf')
    label_confusion(lbl_dict, ref_dict)

    >>>
    p = r'C:/Projects/MachineLearningSpace/labelled_data/'
    date = '201711'
    model_date = '201711'
    ref_dict = mu.read_labels_cdf(os.path.join(p, 'labels_fpi_fast_dis_dist_' + date + '.cdf'))
    lbl_dict = mu.read_labels_cdf(r'C:\Projects\MachineLearningSpace\data\labels_' + model_date + r'_verify01\labels_fpi_fast_dis_dist_' + date + '.cdf')
    label_confusion(lbl_dict, ref_dict)

    """
    assert(np.all(lbl_dict['epoch'] == ref_dict['epoch']))

    # Plotting parameters
    params = {'axes.labelsize': 'medium',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'font.size': 30,
          'font.family': 'sans-serif',
          'text.usetex': False,
          'mathtext.fontset': 'stixsans',}
    plt.rcParams.update(params)

    # First, choose only the 'known' examples
    lbl = lbl_dict['label']
    ref_lbl = ref_dict['label']

    known = np.where(ref_lbl > -1)
    lbl = lbl[known]
    ref_lbl = ref_lbl[known]

    mplt.plot_confusion_matrix(ref_lbl, lbl, np.array([mu.regions[i] for i in range(len(mu.regions)-1)]), normalize=True, title='confusion_matrix_'+date)

    print('Total number of known examples', lbl.shape[0])
    for i in range(len(mu.regions) - 1):
        print(i, 'Actual, human labelled:', mu.regions[i], '| Predicted:')

        human_labelled = np.where(ref_lbl == i)

        for j in range(len(mu.regions) - 1):
            predicted = np.where(lbl[human_labelled] == j)[0]
            print('\t', mu.regions[j], '-', len(predicted))

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    start = time.time()
    # Classify 1 month of data
 #   for model_date in ['201711']:
 #       model_filename = os.path.join(model_path, r'cnn_dis_' + model_date + '_verify.h5')
 #       lbl_path = os.path.join(base_path, r'labels_human' + r'/')
 #       # for year in ['2015', '2016', '2017', '2018', '2019']: # 2017 
 #          # for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
 #       for year in ['2017']:
 #           for month in ['08']:
 #         
 #               date = year + month
 #               # Path to 1 month of FPI data
                # fpi_data_path = os.path.join(data_path, r'mms/' + mms_name + r'/fpi/fast/l2/' + spc_name + r'-dist/' + year + r'/' + month)
                # # FGM data
                # fgm_data_path = os.path.join(data_path, r'mms/' + mms_name + r'/fgm/srvy/l2/' + year + r'/' + month)

                # if os.path.exists(fpi_data_path) and os.path.exists(fgm_data_path):
                #     res_file = classify_one_month(fpi_data_path, lbl_path, model_filename)
    end1 = time.time()
    totalmlcl = end1-start
    totalmlclm = totalmlcl/60
    start3 = time.time()
    # Pass data through autoencoder: get clusters, difference, and compressed representation
    #aut_dict = autoencode_one_month(fpi_data_path, model_filename, n_clusters=6)
    end3 = time.time()
    totalmlc3 = end3-start3
    totalaumin3 = totalmlc3/60
    
    # Check the labels. Also see labels_accuracy().
    lbl_dict = mu.read_labels_cdf(os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + '201712' + '.cdf'))
    #demo_labels(lbl_path, date, lbl_dict=lbl_dict, reg_dict=reg_dict)

    # Compute PCA and pickle it. This should be done on large machine, e.g., Brain, because uses lots of memory.
    start2 = time.time()
    n_components = 16384
    pca_dict = PCA_labelled(fpi_data_path, lbl_dict, n_components=n_components, pic_filename=os.path.join(base_path, 'pca_' + date + '_' + str(n_components) + '.pic'))

    # Plot orbits
    #fgm_dict = mu.read_many_fgm_cdf(fgm_data_path)
    #plot_labels_orbit(fgm_dict, lbl_dict)
    
    pycuda.tools.clear_context_caches()
    end = time.time()
    totals = end - start
    totalm = totals/60
    print("The program took %d seconds or %d minutes" % (totals, totalm))
    print("Classifying took %d seconds or %d minutes" % (totalmlcl, totalmlclm))
    totalsp = end - start2
    totalmp = totalsp/60
    print("PCA took %d seconds or %d minutes" % (totalsp, totalmp))

    pass
