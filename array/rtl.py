'''
Created on Sep 19, 2012

@author: chuxpy

The complete go-to-module on RTL data input, committing, etc.

Data Model:
Sky signal -> Receivers
Receivers+RHTimestamp+AntennaID -> Array
'''

import rtlsdr, datetime, itertools, multiprocessing, tables, sys, time, os, math, scipy, numpy, scipy.signal, scipy.fftpack, scipy.stats

sec, hz = 1.0, 2**20
sample_size = int(sec*hz)
center_freq = 1.4208e9

#metadata; info = tables.Float32Col(shape=3) #essential data for interpretation and shifting of samples: (sample_rate[hz], length[sec])
class sample(tables.IsDescription):
    id = tables.Int32Col(pos=0) #antenna ID
    utcendtime = tables.Time64Col(pos=1) #accurate to milliseconds
    data = tables.Float64Col(shape=sample_size, pos=2) #data from (utcendtime-sec) to (utcendtime) with a step of hz.

class metadata(tables.IsDescription):
    id = tables.Int32Col(pos=0) #antenna ID
    position = tables.Float64Col(shape=2, pos=1) #latitude, longitude
    samplesize = tables.Int64Col(pos=2)
    integrationtime = tables.Float64Col(pos=3)
    sps = tables.Float64Col(pos=4)
    centerfreq = tables.Float64Col(pos=4)

def process_receiver(rtl_queue, rtl_num):
    sdr = rtlsdr.RtlSdr(rtl_num)
    sdr.sample_rate = hz
    sdr.center_freq = center_freq
    sdr.gain = 1
    while True:
        data = sdr.read_samples(sample_size)
        rtl_queue.put({'Receiver': rtl_num, 'Timestamp': time.time(), 'Data': data})

class receiver(object):
    '''Builds and operates the RTL data input circuit.'''
    def __init__(self, rtl_list=[0]):
        self.rtl_list = rtl_list
        self.pairs = list(itertools.combinations(self.rtl_list, 2))
        self.rtl_queue = {key: None for key in self.rtl_list}
        self.rtl_pos = {key: None for key in self.rtl_list}
        for receiver_num in self.rtl_list:
            self.rtl_queue[receiver_num] = multiprocessing.Queue()
            pos_string = raw_input("Please plug in receiver #%s and type in its position x,y as two floats separated by a comma." % receiver_num)
            self.rtl_pos[receiver_num] = tuple([float(num.strip()) for num in pos_string.split(",")][0:1])
            p = multiprocessing.Process(target=process_receiver, args=(self.rtl_queue[receiver_num], receiver_num)) #1st arg to process_receiver() is `self`.
            p.start()
    def get_queue(self):
        return self.rtl_queue
    def initiate_hdf5(self):
        '''Initiates a HDF5 file with the appropriate metadata, and returns a table/array object for writing individual observations (row operations).'''
        now = datetime.datetime.utcnow()
        suffix = 0
        while os.path.isfile('%s-%s-%s-%sh%sm%s.hd5' % (now.year, now.month, now.day, now.hour, now.minute, '_%s' % suffix)): suffix+=1
        h5file = tables.openFile('%s-%s-%s-%sh%sm%s.hd5' % (now.year, now.month, now.day, now.hour, now.minute, '_%s' % suffix), mode='w', title='Dandelion observation file: UTC-ISO: %s' % (datetime.datetime.isoformat(now)))
        group = h5file.createGroup('/', 'detector', 'Detector information')
        table = h5file.createTable(group, 'readout', sample, 'Readout')
        table.cols.utcendtime.createCSIndex()
        metatable = h5file.createTable(group, 'metadata', metadata, 'Metadata')
        for k in self.rtl_pos:
            row = metatable.row
            row['id'] = k
            row['position'] = self.rtl_pos[k]
            row['samplesize'] = sample_size
            row['integrationtime'] = sec
            row['sps'] = hz
            row['centerfreq'] = center_freq
            row.append()
            metatable.flush()
        return table
    def row_write(self, table, rtl_num, data, timestamp):
        '''Writes a row of data to a HDF5 table/array object.'''
        row = table.row
        row['id'] = rtl_num
        row['utcendtime'] = timestamp
        #row['re'], row['im'] = numpy.array([num.real for num in data]), numpy.array([num.imag for num in data])
        #row['data'] = [numpy.sqrt(num.real**2 + num.imag**2) for num in data] #this line does not work
        row['data'] = numpy.abs(data) #is this equivalent?
        row.append()
        table.flush()

def start(length=0, time_limit=0): #stop when length or time limit reached.
    length = int(length)
    current_length = 0
    sources = [0,1,2]
    rc = receiver(sources)
    queue = rc.get_queue()
    table = rc.initiate_hdf5()
    start_time = time.time()
    try:
        while True:
            for (p, q) in queue.items():
                row = q.get() #no timeout
                rc.row_write(table = table, rtl_num = row['Receiver'], data = row['Data'], timestamp = row['Timestamp'])
                if length > 0:
                    current_length += 1
                    if current_length > length:
                        sys.exit('%s entries recorded to the HDF5 table. :) Halted because of a length limit.' % current_length)
                if time_limit > 0:
                    t = time.time() - start_time
                    if t > time_limit:
                        sys.exit('%s seconds passed since start time. :) Halted because of time limit.' % t)
    except SystemExit, exc:
        try:
            table.reIndex()
        finally:
            print exc
            sys.exit()

class correlator(object):
    '''Definition: the correlator has a function that is separated from the receiving logic. It can be executed separately.
    The correlator works by arbitrarily cropping samples and finding their correlation peak, then shifting them and resampling them appropriately
    at fractional delays as defined by the correlation function peak.
    It also uses the measured timestamps (accurate to a millisecond) to calculate the U and V of each baseline, and records it in the MIRIAD
    format for viewing with AIPY.'''
    def read_hdf5(self, filename):
        h5file = tables.openFile(filename, 'r')
        return (h5file.root.detector.readout, h5file.root.detector.metadata)
    def corr_loop_find(self, parent_num, utcendtime, tbl, quality):
        min_overlap = quality*sec
        for num, row in enumerate(tbl.itersorted(tbl.cols.utcendtime, start=parent_num+1), start=parent_num+1):
            if row[1] > (utcendtime + min_overlap): return num
            else: continue
        return parent_num+1

class GetMeOutofThisLoop(BaseException):
    pass

def progress(msg):
    print msg

def crop(child_row, parent_row, time_inacc = 5e-3): #5 millisecond inaccuracy assumed
    child_data, parent_data = child_row['data'], parent_row['data']
    tdiff = parent_row['utcendtime'] - child_row['utcendtime']
    inacc = int(time_inacc*hz)
    offset = int(math.fabs(tdiff) - inacc)
    if offset < 0: offset = 0
    if tdiff > 0:
        child_data = child_data[offset:]
        parent_data = parent_data[0:len(parent_data)-offset]
    elif tdiff < 0:
        child_data = child_data[0:len(child_data)-offset]
        parent_data = parent_data[offset:]
    return (child_data, parent_data)

def correlate_for_true_delay(child_row, parent_row, time_inacc = 5e-3):
    '''The time delay of two signals is given by the arg max of its correlation function. Takes the two rows and returns a normalized time delay.
    Since convolution is similar to but not identical to correlation, it should yield a correct result.
    
    Do an argmax supported by a "pinpointing" and yield a tdiff that is close to the expected tdiff value.'''
    #look at:
    #numpy.argmax() - get index of peak
    child_fft = scipy.fftpack.fft(child_row['data'])
    parent_fft = scipy.fftpack.fft(parent_row['data'])
    child_inverse_conjugate = -child_fft.conjugate()
    #auto_child = numpy.abs(scipy.fftpack.ifft(child_fft*child_inverse_conjugate)) # as per http://hebb.mit.edu/courses/9.29/2002/readings/c13-2.pdf , check for 0's and eliminate them
    #convolution = numpy.abs(scipy.fftpack.ifft(parent_fft*child_inverse_conjugate))
    fft_auto_child = child_fft*child_inverse_conjugate # as per http://hebb.mit.edu/courses/9.29/2002/readings/c13-2.pdf , check for 0's and eliminate them
    fft_convolution = parent_fft*child_inverse_conjugate
    convolution_spectrum = numpy.abs(scipy.fftpack.ifft(fft_convolution/fft_auto_child)) #Roth window
    #we're having troubles actually dealing with the whole convolution spectrum - crop it to several milliseconds from where we think it ought to be to find an accurate tdiff.
    assert time_inacc < sec/2, "This system clock cannot *possibly* be that inaccurate!"
    expected_tdiff = int((child_row['utcendtime']-parent_row['utcendtime'])*hz) #our initial guess for the location of the tdiff peak
    sample_inacc = int(time_inacc*hz) #around the guessed place of tdiff.
    if expected_tdiff-sample_inacc < 0: expected_tdiff = sample_inacc
    elif expected_tdiff+sample_inacc > sample_size: expected_tdiff = sample_size-sample_inacc
    cropped_convolution_spectrum = convolution_spectrum[expected_tdiff-sample_inacc:expected_tdiff+sample_inacc]
    #later measurements of tdiff will have a 0 point at expected_tdiff-sample_inacc and a total length of around 2*sample_inacc (may wrap around)
    tdiff = numpy.argmax(cropped_convolution_spectrum)
    tdiff += expected_tdiff-sample_inacc #check the other case that it wrapped around..?
    zscore = (convolution_spectrum[tdiff]-numpy.average(convolution_spectrum))/numpy.std(convolution_spectrum)
    #timespan = math.fabs(child_row['utcendtime'] - parent_row['utcendtime']) + sec
    expected_overlap = (float(sec) - 1.0*math.fabs(child_row['utcendtime']-parent_row['utcendtime']))*hz
    real_overlap = (float(sec) - 1.0*math.fabs(float(tdiff)/hz))*hz
    logmsg = '(%s,%s)| peak: (%s,%s), error: %s s, real: %s, expected: %s, z-score: %s'
    print  logmsg %(parent_row['id'],
                    child_row['id'],
                    tdiff,
                    convolution_spectrum[tdiff],
                    (float(tdiff)/hz-1.0*(child_row['utcendtime']-parent_row['utcendtime'])),
                    real_overlap,
                    expected_overlap,
                    round(zscore,3))
    return tdiff

def resample(row, var_delay):
    return (row, true_crop_dist)

def b_filter(data, low, high, sample_rate, order=3, filter_type='band'):
    '''Defines a Butterworth bandpass filter defined by args. Applies filter to an input and returns it.
    filter_type specifies the type of filter used.'''
    nyquist = 0.5 * sample_rate
    filter_coeff = [low / nyquist, high / nyquist]
    if filter_type == 'low':
        filter_coeff = [high / nyquist]
    elif filter_type == 'high':
        filter_coeff = [low / nyquist]
    b, a = scipy.signal.butter(order, filter_coeff, btype=filter_type)
    return scipy.signal.lfilter(b, a, data)

def get_uv_rate(timearray, rtlids):
    pass

def correlate(filename, quality=0.5): #quality denotes the minimum permissible overlap between two samples
    assert quality <= 1.0+1e-21, "The quality of correlation cannot be over 100%."
    c = correlator()
    tbl, meta = c.read_hdf5(filename)
    detector_metadata = {}
    for row in meta:
        detector_metadata[row['id']] = {'position': row['position'],
                                        'samplesize': row['samplesize'],
                                        'integrationtime': row['integrationtime'],
                                        'sps': row['sps'],
                                        'centerfreq': row['centerfreq']}
    for num, row in enumerate(tbl.itersorted(tbl.cols.utcendtime)):
        min_overlap = quality*sec
        iterated = enumerate(tbl.itersorted(tbl.cols.utcendtime, start=num, stop=tbl.nrows), start=num)
        try:
            anum, arow = iterated.next()
            while True:
                if not arow['utcendtime'] > (row['utcendtime'] + min_overlap):
                    #do all processing and passing here!
                    child_dict, parent_dict = {}, {}
                    #child_dict['data'], parent_dict['data'] = crop(arow, row)
                    child_dict['data'], parent_dict['data'] = b_filter(arow['data'], 0, 2**19, 2**20, filter_type='low'), b_filter(row['data'], 0, 2**19, 2**20, filter_type='low')
                    child_dict['utcendtime'], parent_dict['utcendtime'] = arow['utcendtime'], row['utcendtime']
                    child_dict['id'], parent_dict['id'] = arow['id'], row['id']
                    var_delay = correlate_for_true_delay(child_dict, parent_dict)
                    #child_dict['data'], parent_dict['data'] = resample(child_dict['data'], var_delay)
                    anum, arow = iterated.next()
                else:
                    raise GetMeOutofThisLoop
        except StopIteration: #set stop index pls
            anum += 1
        except GetMeOutofThisLoop: #the stop index does not need to be set
            pass
        all_row_indices = range(num, anum)
        #print '%s: %s. %s' %(num, all_row_indices, row[1]) #this is not part of final

#correlate('2012-9-30-22h19m_0.hd5')
