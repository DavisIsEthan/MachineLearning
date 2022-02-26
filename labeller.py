""" Labeller routines for the observations of MMS FPI DIS ion spectrometer.

(c) Vyacheslav Olshevsky (2019)

"""
import numpy as np
import mms_utils as mu
import mms_plots as mplt
import gui_utils
from glob import glob, iglob
import os, shutil, cdflib
import matplotlib.pyplot as plt
import datetime, pickle, collections
import matplotlib.dates as mdates
from sklearn.utils import shuffle
import  pycuda.autoprimaryctx


# Some constants
data_path = r'C:\\Users\\Davis\\Desktop\\seniorresearch\\volshevsky-mmslearning-davisWIP\\data\\' 
base_path = r'C:\\Users\\Davis\\Desktop\\seniorresearch\\volshevsky-mmslearning-davisWIP\\labels_human\\' 
year = r'2017'
month = r'11'
ref_date = year + month

# [mms1, mms2, mms3, mms4]
mms_index = 1
mms_name = r'mms' + str(mms_index)

# [dis, des]
spc_name = r'dis'
var_name = mms_name + '_' + spc_name + '_dist_fast'
labelled_path='C:\\Users\\Davis\\Desktop\\seniorresearch\\volshevsky-mmslearning-davisWIP\\labels_human'
reference_path = os.path.join(labelled_path, 'reference_' + spc_name + '.pic')
fpi_data_path = os.path.join(data_path, r'mms\\' + mms_name + r'\\fpi\\fast\\l2\\' + spc_name + r'-dist\\' + year + r'\\' + month)
# FGM data
fgm_data_path = os.path.join(data_path, r'mms\\' + mms_name + r'\\fgm\\srvy\\l2\\' + year + r'\\' + month)
fgm_dict = mu.read_many_fgm_cdf(fgm_data_path)

with open(r'C:\\Users\\Davis\\Desktop\\seniorresearch\\volshevsky-mmslearning-davisWIP\\pca_' + ref_date[:6] + '.pic', 'rb') as f:
    pca_dict = pickle.load(f)





def lbl_name_for_fpi(fpi_filename):
    """ Generates a name of the CDF file with labels given the FPI filename.

    """
    d = mu.parse_mms_filename(fpi_filename)
    return r'_'.join(['labels', d['instrument'], d['tmMode'], d['detector'], d['product'], d['timestamp'][:6]]) + r'.cdf'

def quick_plot(i, energy, dist, epoch, ranges='shock'):
    """ A quick skymap given index.

    ranges: 'shock', 'sheath'

    """
    if ranges == 'shock':
        # The default plotting ranges: SW, Bow shock, Ion foreshock
        mplt.plot_skymap(str(i), energy[0], dist[i])
    elif ranges == 'sheath':
        # Higher energy plots: magnetsheath, magnetosphere.
        mplt.plot_skymap(str(i), energy[0], dist[i], channels=[12, 17, 20, 25, 27, 30], subplot_ranges=np.power(10, [[-23., -20.], [-24., -21.], [-25., -22.], [-26., -23.], [-27., -24.], [-29., -26.]]))

def write_one_label(filename, var_dict):
    """ Write one file with labelled data.
    Currently, pickle.

    """
    print('Writing:', filename)

    if not os.path.exists(os.path.split(filename)[0]):
        os.makedirs(os.path.split(filename)[0])

    with open(filename, 'wb') as pic_file:
        pickle.dump(var_dict, pic_file)

def load_one_label(filename):
    """ Load one file with labelled data and return an array with labels.

    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict['labels']
    return None

def compute_ids(ref_dict):
    """ Compute unique IDs for examples using epoch and mms index.

    """
    mms_id_choices = [mu.parse_mms_filename(f)['mmsId'] for f in ref_dict['filenames']]
    #print(mms_id_choices)
    fileids = ref_dict['file_id']
    mmsids = np.ones(fileids.shape)
    for i in range(len(mms_id_choices)):
        mmsids[np.where(fileids == i)] = mms_id_choices[i]
    #mms_ids = ref_dict['file_id'].choose(mms_id_choices)
    uids = mu.epoch2id(ref_dict['epoch'], mmsids)
    return uids

def compute_recommended_labels(dist, ref_classes):
    """ Compute distances to each of reference volumes.

    """
    print('Computing distances to reference regions')
    lgdist = mu.compute_dist_log(dist)

    keys = sorted(ref_classes.keys())
    ref_distances = np.empty((dist.shape[0], len(keys)))
    for i in range(len(keys)):
        c = keys[i]
        print(mu.regions[c])
        ref_distances[:, i] = ((lgdist - ref_classes[c])**2).sum(axis=(1,2,3))**0.5

    recommended_classes = ref_distances.argmin(axis=1)
    recommended_classes = recommended_classes.choose(keys)

    return recommended_classes

def irfu_recommended_labels(epoch):
    """ Read intervals labelled by IRFU and map them onto the given moments.

    """
    # TODO: this looks dirty, but txt files themselves contain wrong data (data for wrong dates). 
    # It seems that  mms1_edp_sdp_regions_20171206_v0.0.0.txt is reasonable.
    reg_dict = mu.read_irfu_regions(os.path.join(data_path, r'irfu/cal/mms1/edp/sdp/regions/mms1_edp_sdp_regions_20171206_v0.0.0.txt'))
    # Get regions for labelled data. Note that IRFU doesn't distinguish between q-par and 1-perp sheath.
    irfu_labels = np.vectorize(mu.irfu_to_my_regions.get)(reg_dict['region'])

    # Interpolate region periods on epochs
    return mu.map_interval_labels(irfu_labels, reg_dict['epoch'], epoch)

def helper_plots(fpi_dict, labels, fgm_dict, recommended_labels=[], ref_dict={}, pca_dict={}):
    """ Produce helper plots for classification of FPI DIS obervations.

    """
    print('Making helper plots, it might take a few minutes...')
    #print(fgm_dict)
    epoch_fgm = fgm_dict['epoch']
    B_GSM = fgm_dict[mms_name + '_fgm_b_gsm_srvy_l2']

    dist, epoch_fpi = fpi_dict['dist'], fpi_dict['epoch'], 
    indices = np.arange(epoch_fpi.shape[0])

    # Common time axis plot ranges
    epoch_lim = (np.min([epoch_fgm.min(), epoch_fpi.min()]), np.max([epoch_fgm.max(), epoch_fpi.max()]))
    epoch_lim = [epoch_fpi.min(), epoch_fpi.max()]
    time_fpi = mu.epoch2time(epoch_fpi)
    time_lim = [mdates.date2num(mu.epoch2time(epoch_lim[0])), mdates.date2num(mu.epoch2time(epoch_lim[1]))]

    # 1. Plot ion distribution averaged over phi, theta.
    fig = plt.figure('Plots for ' + time_fpi[0].isoformat()[:10].replace('-', '') + '_' + mms_name, figsize=(38, 19))
    ax1 = fig.add_subplot(4, 1, 1)
    mplt.plot_avg_dist(dist, time_fpi, fpi_dict['energy'], ax1)
    ####print(epoch_lim, epoch_fgm)
    # 2. Plot B field in GSM
    ax2 = fig.add_subplot(4, 1, 2)
    B_lbl = {0: 'X', 1: 'Y', 2: 'Z', 3: '|B|'}
    # Auto adjust vertical axis limits by plotting only a piece within epoch limits
    fgm_lim = [max(epoch_lim[0], epoch_fgm[0]), min(epoch_lim[1], epoch_fgm[-1])]
    plt_slice = np.where(np.logical_and((epoch_fgm >= fgm_lim[0]), (epoch_fgm <= fgm_lim[1])))
    fgm_t = mu.epoch2time(epoch_fgm[plt_slice])
    ax2.plot(fgm_t, B_GSM[:,3][plt_slice], label=B_lbl[3], linewidth=2)
    ax2.set_ylabel('|B| GSM [nT]')

    # 3. Plot some characteristic of the distribution, like, e.g. variance
    ax3 = fig.add_subplot(4, 1, 3)
    r = fgm_dict['mms1_fgm_r_gse_srvy_l2'] / mu.earth_radius
    re = fgm_dict['epoch_state']
    r_lbl = {0: 'X', 1: 'Y', 2: 'Z', 3: '|r|'}
    r_lim = [max(epoch_lim[0], re[0]), min(epoch_lim[1], re[-1])]
    plt_slice = np.where(np.logical_and((re >= r_lim[0]), (re <= r_lim[1])))
    ax3.plot(mu.epoch2time(re[plt_slice]), r[:,3][plt_slice], label=r_lbl[3], linewidth=2)
    ax3.set_ylabel('|r| GSE [Re]')

    '''
    # Plot distance to ref classes
    lgdist = mu.compute_dist_log(dist) #np.ma.log10(dist)
    stdd = np.mean(lgdist, axis=(1, 2, 3))
    ax3.plot(time_fpi, stdd, linewidth=2)    
    ax3.set_ylabel('<lg(Dist)>')
    '''
    # Plot first PCA component in the twin Y axis.
    if len(pca_dict) > 0:
        epoch_pca = pca_dict['epoch']
        pca0 = pca_dict['projection']

        pca_lim = [max(epoch_lim[0], epoch_pca[0]), min(epoch_lim[1], epoch_pca[-1])]
        plt_slice = np.where(np.logical_and((epoch_pca >= pca_lim[0]), (epoch_pca <= pca_lim[1])))
        pca_t = mu.epoch2time(epoch_pca[plt_slice])

        axt = ax3.twinx()
        axt.set_ylabel('PCA #0')
        axt.plot(pca_t, pca0[plt_slice], linewidth=2, color='darkolivegreen')

    # 4. Plot the label
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(time_fpi, labels, label='Labelled', linewidth=2)

    # Labels 'recommended' by shortest distance
    if len(recommended_labels) > 0:
        ax4.scatter(time_fpi, recommended_labels, s=20, c='slategrey', label='Recommended', marker='o')

    # Plot reference labels, if any are present in this interval
    if len(ref_dict) > 0:
        ax4.scatter(ref_dict['epoch'], ref_dict['label'], s=100, c='firebrick', marker='o', label='Reference')

    yticks = np.array(list(mu.regions.keys()))
    ax4.set_ylim(yticks.min()-0.5, yticks.max()+0.5)
    ax4.set_yticks(yticks)
    ax4.set_yticklabels([mu.regions[k] for k in mu.regions.keys()])
    ax4.legend(loc=0)
    ax4.grid(axis='y')

    # Fix time annotation
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(time_lim)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # This simply sets the x-axis data to diagonal so it fits better.
    #fig.autofmt_xdate()
        
    # Add snapshot index grid for convenience
    di = float(time_lim[1] - time_lim[0]) / time_fpi.shape[0]
    t0 = mdates.date2num(time_fpi[0])
    index_lim = ((time_lim[0] - t0) / di, (time_lim[1] - t0) / di)
    try:
        index_ticks = [s.stop for s in indices]
    except:
        index_ticks = indices
    time_ticks = [di * i + mdates.date2num(time_fpi[0]) for i in index_ticks]

    for ax in [ax1, ax2, ax3, ax4]:
        #for x in time_ticks:
        #    ax.axvline(x=x, linewidth=1, color='dimgrey')
        # Add twin axis with time/index
        axt = ax.twiny()
        #axt.set_xlabel('Snapshot index', labelpad=0)
        axt.set_xlim(index_lim)
        axt.tick_params(axis="x", direction="in", pad=0)

    plt.tight_layout()
    plt.show()
    return fig

def clean_filenames(ref_dict):
    """ Removes a filename from list which is a part of ref_dict.
    
    i - the index of the fnames list item to be removed. 
    fid - array of references to items from fnames

    """
    fnames = ref_dict['filenames']
    fid = ref_dict['file_id']
    i = 0
    while i < fnames.shape[0]:
        print(i, fnames[i])
        if np.all(fid != i):
            fid = np.where(fid > i, fid - 1, fid)
            fnames = np.delete(fnames, [i,])
        else:
            i += 1
    return fnames, fid

def stopcontinue(func):
    """ A decorator to ask whether to stop or to continue processing.

    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        choice = False if input('Press Enter to continue or input "N" to stop ').upper() == 'N' else True
        return result, choice

    return wrapper

@stopcontinue
def run_labeller_gui(labels, fpi_dict, fgm_dict, recommended_labels=[], pca_dict={}):
    """ Makes demo plots and allows to edit labels, without saving them anywhere.

    """
    def gui_plot(i):
        ranges = 'sheath' if labels[i] in [2, 3, 4] else 'shock'
        quick_plot(i, fpi_dict['energy'], fpi_dict['dist'], fpi_dict['epoch'], ranges=ranges)

    def gui_plotall():
         return helper_plots(fpi_dict, labels, fgm_dict, recommended_labels=recommended_labels, pca_dict=pca_dict)

    return gui_utils.edit_array(labels, plot_item=gui_plot, plot_all=gui_plotall)

def create_new_labels_file(lbl_path, fpi_filename):
    """ Create a CDF file with labels
    Generate file name for the labelled data if needed

    """
    existing_lbl_file = None
    print(lbl_path)
    if os.path.isfile(lbl_path):
        lbl_filename = lbl_path
        # Rename existing file to a temporary one
        existing_filename = os.path.splitext(lbl_filename)[0] + '.tmp.cdf'
        print('Renaming', lbl_filename, existing_filename)
        os.rename(lbl_filename, existing_filename)
        existing_lbl_file = cdflib.cdfread.CDF(existing_filename)
    else:
        if os.path.isdir(lbl_path):
            lbl_filename = os.path.join(lbl_path, lbl_name_for_fpi(fpi_filename))
            if os.path.isfile(lbl_filename):
                # Rename existing file to a temporary one
                existing_filename = os.path.splitext(lbl_filename)[0] + '.tmp.cdf'
                print('Renaming', lbl_filename, existing_filename)
                os.rename(lbl_filename, existing_filename)
                existing_lbl_file = cdflib.cdfread.CDF(existing_filename)
            else:
                print('Generated new file name in existing folder', lbl_filename)
        else:
            if lbl_path.endswith('.cdf'):
                os.makedirs(os.path.split(lbl_path)[0], exist_ok=True)
                lbl_filename = lbl_path
                print('Created new file given its name', lbl_filename)
            else:
                os.makedirs(lbl_path)
                lbl_filename = os.path.join(lbl_path, lbl_name_for_fpi(fpi_filename))
                print('Generated file name and created new file', lbl_filename)

    print('Reading', lbl_filename)
    if existing_lbl_file != None:
        print('Existing', existing_lbl_file.file)

    # Create an empty CDF file for labels
    cdf_spec = {'Copyright': 'Vyacheslav Olshevsky (slavik@kth.se)', 'Majority': 'Column_major', 'Encoding': 6, 'Checksum': True, 'rDim_sizes': [], 'Compressed': False}
    lbl_cdf_file = cdflib.cdfwrite.CDF(lbl_filename, cdf_spec=cdf_spec, delete=False)

    return lbl_cdf_file, existing_lbl_file

def append_to_cdf(new_cdf, existing_cdf):
    """ Copy all zVariables from an existing CDF file
    into a new one, if they are absent in the new one.

    """
    print('Copying variables from existing CDF, this might take a while..')
    ex_info = existing_cdf.cdf_info()
    for v in ex_info['zVariables']:
        if not (v in new_cdf.zvars):
            print('Copying old variable', v)
            varinfo = existing_cdf.varinq(v)
            varattrs = existing_cdf.varattsget(v, expand=True)
            vardata = existing_cdf.varget(v)
            new_cdf.write_var(varinfo, var_attrs=varattrs, var_data=vardata)

def fpi_filenames_generator_folder(fpi_path, mms_name, spc_name, start_index=0, verbose=True):
    """ Yields FPI CDF filenames from the specified folder.

    """
    if not(os.path.exists(fpi_path)):
        raise FileExistsError('The specified fpi_path does not exist!\n'+fpi_path)

    names = glob(os.path.join(fpi_path, mms_name + r'*fpi_fast_l2_' + spc_name + r'-dist_*.cdf'))
    if len(names) < 1:
        raise FileNotFoundError('No matching files in fpi_path!')
    names.sort()

    for i in range(start_index, len(names)):
        name = names[i]
        if verbose:
            print('\n%i/%i %s' % (i, len(names), os.path.basename(name)))
        yield name

def fpi_filenames_generator_list(names, start_index=0, verbose=False):
    """ Yields FPI CDF filenames given a list of suffixes read from the human-made labels CDF.
    They have the following form: 
        _mms1_fpi_fast_dis_dist_20171130220000

    Parameters:
        names - a list of names in the above form

    """
    names.sort()

    for i in range(start_index, len(names)):
        name = names[i]
        parts = name.strip('_').split('_')

        folder = os.path.join(data_path, parts[0] + r'/fpi/fast/l2/' + parts[3] + r'-dist/' + parts[5][:4] + r'/' + parts[5][4:6])
        fname = os.path.join(folder, parts[0] + '_' + parts[1] + '_' + parts[2] + '_l2_' + parts[3] + '-' + parts[4] + '_' + parts[5] + '_v3.3.0.cdf')

        if not(os.path.exists(fname)):
            raise FileExistsError('The following file is missing:\n' + fname + '\ndata_path=', data_path)

        if verbose:
            print('\n%i/%i %s' % (i, len(names), os.path.basename(fname)))

        yield fname

def fpi_data_extractor(fpi_filepath, lbl_file, fgm_dict, start_index=0, verbose=True):
    """ Given fpi_filename, extracts fpi_dict, fgm_dict and labels.

    """
    # Read energy distribution fom CDF file
    fpi_dict = mu.read_fpi_cdf(fpi_filepath)

    # Read existing labels or generate new ones
    var_suffix = mu.get_label_fpi_suffix(fpi_filepath)
    if (lbl_file != None) and (lbl_file.varattsget('label' + var_suffix)):
        # Read labels from existing file
        labels = lbl_file.varget('label' + var_suffix)
    else:
        # Label as Unknown
        labels = np.zeros((fpi_dict['epoch'].shape[0],), dtype=np.int8) - 1

    # Read new FGM data if needed
    fgm_dict = mu.read_many_fgm_cdf(fgm_data_path)

    return fpi_dict, fgm_dict, labels

def label_multiple_cdf(filenames, lbl_path, start_index=0, pca_dict={}):
    """ CDF files with FPI data contain 2 hours of data or less.
    We will write 1 CDF file with labels per month (1 folder with CDF files in MMS catalogue).

    Since CDFLib doesn't know how to append variables or modify variables in-place,
    we shall create a new CDF file and copy all data from the existing one.

    # Parameters:
        fpi_path - path to the specific month in the MMS catalogue.
        lbl_path - the file or folder where to save the labels.
                   The filename is generated automatically if only the folder name is given.

    # Example:

    label_multiple_cdf([path + 'mms1_fpi_fast_l2_dis-dist_20171201000000_v3.3.0.cdf', 
                        path + 'mms1_fpi_fast_l2_dis-dist_20171201080000_v3.3.0.cdf',], 
                        labelled_path)

    """
    # Create a new file. If necessary, move existing to temporary.
    existing_lbl_file = None
    new_lbl_file = None


    
    # Because FGM data is stored on a daily basis, we recycle fgm_dict for multiple FPI files
    fgm_dict = None
    filenameslist = list(filenames) #### Otherwise you get a generator object error on the next line
    for fpi_filepath in filenameslist[start_index:]:
        if new_lbl_file is None:
            new_lbl_file, existing_lbl_file = create_new_labels_file(lbl_path, fpi_filepath)

        fpi_dict, fgm_dict, labels = fpi_data_extractor(fpi_filepath, existing_lbl_file, fgm_dict)

        # Show GUI to modify the labels
        labels, choice = run_labeller_gui(labels, fpi_dict, fgm_dict, recommended_labels=[], pca_dict=pca_dict)
    
        # Set probability to 1 for all human-made labels.
        prob = np.zeros((labels.shape[0], len(mu.regions)-1), dtype=np.float32)
        prob[np.arange(labels.shape[0]), labels] = 1.0

        print(labels.shape, labels.dtype, prob.shape, prob.dtype)
        mu.add_labels_CDF(fpi_dict['epoch'], labels, prob, fpi_filepath, new_lbl_file)
        new_lbl_file.close()

        if not choice:
            break

    # Copy the remaining vars and delete temporary file
    if existing_lbl_file != None:
        append_to_cdf(new_lbl_file, existing_lbl_file)
        existing_lbl_file.close()
        os.remove(existing_lbl_file.file)
        labels, choice = run_labeller_gui(labels, fpi_dict, fgm_dict)


def label_one_month(fpi_path, lbl_path, start_index=0, pca_dict={}):
    """ A shortcut to label all files for 1 month of DIS observations.

    # Parameters:
        fpi_path - path to the specific month in the MMS catalogue.
        lbl_path - the file or folder where to save the labels.
                   The filename is generated automatically if only the folder name is given.

    # Example:

    with open(r'C:/Projects/MachineLearningSpace/data/pca_' + ref_date[:6] + '.pic', 'rb') as f:
        pca_dict = pickle.load(f)
    label_one_month(fpi_data_path, labelled_path, start_index=80, pca_dict=pca_dict)

    """
    return label_multiple_cdf(fpi_filenames_generator_folder(fpi_path, mms_name, spc_name, start_index=start_index, verbose=False), 
                              lbl_path, start_index=start_index, pca_dict=pca_dict)

def demo_one_month_labels(fpi_path, lbl_path, start_index=0, pca_dict={}):
    """ Show plots for a month of labelled data.

    # Example:

    with open(r'C:/Projects/MachineLearningSpace/data/pca_' + ref_date[:6] + '.pic', 'rb') as f:
        pca_dict = pickle.load(f)
    demo_one_month_labels(fpi_data_path, r'C:/Projects/MachineLearningSpace/labelled_data/labels_fpi_fast_dis_dist_201711.cdf', start_index=10, pca_dict=pca_dict)

    """
    # Create a new file. If necessary, move existing to temporary.
    lbl_file = cdflib.cdfread.CDF(lbl_path)
    
    # Because FGM data is stored on a daily basis, we recycle fgm_dict for multiple FPI files
    fgm_dict = None

    for fpi_filepath in fpi_filenames_generator_folder(fpi_path, mms_name, spc_name, start_index=start_index):
        fpi_dict, fgm_dict, labels = fpi_data_extractor(fpi_filepath, lbl_file, fgm_dict)

    

        if not choice:
            break

    lbl_file.close()
        # Show GUI to modify the labels
    labels, choice = run_labeller_gui(labels, fpi_dict, fgm_dict) #, recommended_labels=irfu_recommended_labels(fpi_dict['epoch']), pca_dict=pca_dict)

def prepare_new_train_set(fraction=0.1, year='2017', month='11', choice_prob={-1: 0.0, 0: 0.25, 1: 1.0, 2: 0.3, 3: 1.0},
                          subtract_err=True):
    """ Prepare a train set from a month of labelled data.
    In the human-labelled datasets for 201711 and 201712 the numbers of examples are imbalanced, 
    therefore we shall disregard ~2/3 of SW samples and ~1/2 of Magnetosheath.
    This is controlled by keyword choice_prob.

    # Example:

    ref_dict = prepare_new_train_set(fraction=0.1, year='2017', month='11')

    """
    np.random.seed(1)
    lbl_path = r'C:/Projects/MachineLearningSpace/labelled_data'
    # Path to the database with MMS data
    data_path = r'C:/Projects/MachineLearningSpace/data'
    lbl_filename = os.path.join(lbl_path, r'labels_fpi_fast_dis_dist_' + year + month + '.cdf')

    train_filename = os.path.splitext(lbl_filename)[0] + '_train.cdf'
    if os.path.exists(train_filename):
        raise FileExistsError('Error! The output file already exists, please rename it\n', train_filename)

    # Read all epochs and labels at once to gather statistics and adjust choice probabilities
    ref_dict = mu.read_labels_cdf(lbl_filename)
    ref_labels, ref_epoch = ref_dict['label'], ref_dict['epoch']
    total_examples = ref_labels.shape[0]
    chosen_examples = int(fraction*total_examples)

    # Choose the requested fraction of all examples with given choice probabilities
    prob = np.vectorize(choice_prob.get)(ref_labels)
    prob /= prob.sum()
    chosen = np.random.choice(np.arange(total_examples, dtype=np.int32), size=chosen_examples, p=prob)
    chosen = np.sort(np.unique(chosen))
    chosen_examples = chosen.shape[0]

    # Prepare empty arrays for the train/test dataset
    Y = ref_labels[chosen].astype(np.int8)
    assert(np.all(Y > -1))
    X_epoch = ref_epoch[chosen]
    X = np.empty((chosen_examples, 32, 16, 32), dtype=np.float32)

    mu.describe_dataset(Y)
    print(os.path.splitext(lbl_filename)[0] + '_train.cdf')
    if input('Enter any key to continue, N to stop ').upper() == 'N':
        return None

    # Read FPI observations one-by-one and copy the dist values for chosen epochs
    for fpi_filename in fpi_filenames_generator_list(ref_dict['files'], start_index=0):
        fpi_dict = mu.read_fpi_cdf(fpi_filename, varname=var_name, subtract_err=subtract_err)
        fpi_dist, fpi_epoch = fpi_dict['dist'], fpi_dict['epoch']
        _, ind_ref, ind_fpi = np.intersect1d(X_epoch, fpi_epoch, return_indices=True, assume_unique=True)

        if len(ind_ref) > 0:
            X[ind_ref] = fpi_dist[ind_fpi]

        print('%i examples added' % (ind_ref.shape[0],))

    X, X_epoch, Y = shuffle(X, X_epoch, Y)

    ref_dict = {'epoch': X_epoch, 'label': Y, 'dist': X, 'id': mu.epoch2id(X_epoch, mms_name),}
    mu.write_dict_cdf(train_filename, ref_dict)

    return ref_dict

if __name__ == '__main__':

  #  plt.ion()

    # 1. Run this to create a CDF with reference labels
    label_one_month(fpi_data_path, labelled_path, start_index=80)

    # 2. Check a month of labels
  #  demo_one_month_labels(fpi_data_path, r'/Users/davis/Downloads/volshevsky-mmslearning-e4c932a93503/labels_human/labels_fpi_fast_dis_dist_201711.cdf', start_index=10)
    
    # 3. This is done after a month of data has been labelled, to prepare for simple_learning
  #  ref_dict = prepare_new_train_set(fraction=0.1, year='2017', month='11', subtract_err=False)
  

    # 4. Manual labelling 
    label_multiple_cdf(fpi_filenames_generator_folder(fpi_data_path, mms_name, spc_name, start_index=0, verbose=False), labelled_path, start_index=0, pca_dict=pca_dict)

    pass