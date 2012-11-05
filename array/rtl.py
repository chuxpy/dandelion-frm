'''
Created on Sep 19, 2012

@author: chuxpy

The complete go-to-module on RTL data input, committing, etc.

Data Model:
Sky signal -> Receivers
Receivers+RHTimestamp+AntennaID -> Array
'''

import rtlsdr, datetime, itertools, multiprocessing, tables, sys, time, os, math, scipy, numpy, scipy.signal, scipy.fftpack, scipy.stats, matplotlib.pyplot as plt, ephem
from numexpr import evaluate

sec, hz = 1.0, 2**20
sample_size = int(sec*hz)
interest_freq = 1.4208e9
dc_width = 1.0e5
center_freq = interest_freq - dc_width

#metadata; info = tables.Float32Col(shape=3) #essential data for interpretation and shifting of samples: (sample_rate[hz], length[sec])
class sample(tables.IsDescription):
    id = tables.Int32Col(pos=0) #antenna ID
    utcendtime = tables.Time64Col(pos=1) #accurate to milliseconds
    data = tables.Float64Col(shape=sample_size, pos=2) #data from (utcendtime-sec) to (utcendtime) with a step of hz.

class metadata(tables.IsDescription):
    id = tables.Int32Col(pos=0) #antenna ID
    position = tables.Float64Col(shape=2, pos=1) #from zero

class fringe(tables.IsDescription):
    id = tables.Int32Col(shape=2, pos=0) #antenna IDs (baseline)
    omega = tables.Float64Col(pos=1)
    xcoeff = tables.Float64Col(pos=2)
    ycoeff = tables.Float64Col(pos=3)
    zscore = tables.Float64Col(pos=4)
    utcmidtime = tables.Time64Col(pos=5)

def process_receiver(rtl_queue, sdr, rtl_num, gain, highfreq):
    print "* %s process spawned" % rtl_num
    sdr.sample_rate = hz
    sdr.center_freq = center_freq
    sdr.gain = gain
    while True:
        data = sdr.read_samples(sample_size)
        fdata = numpy.abs(b_filter(data, dc_width, highfreq, hz, filter_type='band'))
        rtl_queue.put({'Receiver': rtl_num, 'Timestamp': time.time(), 'Data': fdata})

class receiver(object):
    '''Builds and operates the RTL data input circuit.'''
    def __init__(self, rtl_list=[0], gain=10, highfreq=dc_width+100e3):
        self.rtl_list = rtl_list
        self.pairs = list(itertools.combinations(self.rtl_list, 2))
        self.rtl_queue = {key: None for key in self.rtl_list}
        self.rtl_pos = {key: None for key in self.rtl_list}
        self.rtl_rcvrs = {key: None for key in self.rtl_list}
        for receiver_num in self.rtl_list:
            self.rtl_queue[receiver_num] = multiprocessing.Queue()
            pos_string = raw_input("Plug in receiver #%s and type in its position x,y in wavelengths. " % receiver_num)
            self.rtl_pos[receiver_num] = tuple([float(num.strip()) for num in pos_string.split(",")][0:2])
            self.rtl_rcvrs[receiver_num] = rtlsdr.RtlSdr(receiver_num)
        for receiver_num in self.rtl_list:
            p = multiprocessing.Process(target=process_receiver, args=(self.rtl_queue[receiver_num], self.rtl_rcvrs[receiver_num], receiver_num, gain, highfreq)) #1st arg to process_receiver() is `self`.
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
        h5file.root._v_attrs.samplesize = sample_size
        h5file.root._v_attrs.integrationtime = sec
        h5file.root._v_attrs.sps = hz
        h5file.root._v_attrs.centerfreq = center_freq
        h5file.root._v_attrs.latlong = numpy.array([raw_input('Latitude in degrees: '), raw_input('Longitude in degrees: ')])
        metatable = h5file.createTable(group, 'metadata', metadata, 'Metadata')
        for k in self.rtl_pos:
            row = metatable.row
            row['id'] = k
            row['position'] = self.rtl_pos[k]
            row.append()
            metatable.flush()
        return table
    def row_write(self, table, rtl_num, data, timestamp):
        '''Writes a row of data to a HDF5 table/array object.'''
        row = table.row
        row['id'] = rtl_num
        row['utcendtime'] = timestamp
        row['data'] = data
        row.append()
        table.flush()

def start(length=0, time_limit=0, gain=10, highfreq=hz/2): #stop when length or time limit reached.
    length = int(length)
    current_length = 0
    sources = [int(x) for x in raw_input("Please list the rtl_ids of your receivers: ").split(",")]
    rc = receiver(sources, gain, highfreq)
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

class correlator(object):
    def __init__(self):
        self.obs_site = ephem.Observer()
        self.secs_in_sidereal_day = 86164.09054 #wolframalpha
        self.quality = 0.5
        self.time_inacc = 1e-2
    def read_hdf5(self, filename):
        self.h5file = tables.openFile(filename, 'r')
        self.filename = filename
        self.samplesize = self.h5file.root._v_attrs.samplesize #automatically read metadata and load it
        self.integrationtime = self.h5file.root._v_attrs.integrationtime
        self.sps = self.h5file.root._v_attrs.sps
        self.centerfreq = self.h5file.root._v_attrs.centerfreq
        self.latlong = self.h5file.root._v_attrs.latlong
        self.obs_site.lat, self.obs_site.long = [str(attribute) for attribute in self.latlong] #continue setting up obs_site for sidereal stuff
        self.detector_pos = {}
        self.parent_row, self.child_row = {}, {}
        for row in self.h5file.root.detector.metadata:
            self.detector_pos[row['id']] = row['position']
    def next_row(self):
        for num, row in enumerate(self.h5file.root.detector.readout.itersorted(self.h5file.root.detector.readout.cols.utcendtime)):
            min_overlap = self.quality*self.integrationtime
            self.parent_row = {'data': row['data'],
                               'utcendtime': row['utcendtime'],
                               'id': row['id']}
            for anum, arow in enumerate(self.h5file.root.detector.readout.itersorted(self.h5file.root.detector.readout.cols.utcendtime, start=num+1, stop=self.h5file.root.detector.readout.nrows), start=num+1):
                    if not arow['utcendtime'] > (row['utcendtime'] + min_overlap):
                        self.child_row = {'data': arow['data'],
                                          'utcendtime': arow['utcendtime'],
                                          'id': arow['id']}
                    else: break
                    yield ((num, anum),(self.parent_row['id'],self.child_row['id']))
    def initiate_hdf5(self, h5class = fringe):
        '''Initiates a HDF5 file and returns a table corresponding to a number of fringe-rate map lines.'''
        suffix = 0
        now = datetime.datetime.utcnow()
        while os.path.isfile('dFRM_%s%s.hd5' % (self.filename[0:len(self.filename)-4], '_%s' % suffix)): suffix+=1
        h5file = tables.openFile('dFRM_%s%s.hd5' % (self.filename[0:len(self.filename)-4], '_%s' % suffix), mode='w', title='Dandelion fringe rate mapping file: UTC-ISO: %s' % (datetime.datetime.isoformat(now)))
        group = h5file.createGroup('/', 'correlator', 'Correlated information')
        self.save_table = h5file.createTable(group, 'readout', h5class, 'Readout')
        self.save_table.cols.utcmidtime.createCSIndex()
    def correlate(self):
        '''Estimates the time-delay in whole samples, then aligns and finds the complex visibility for two signals in the data key of self.parent_row and self.child_row.
        Generates self.convolution_spectrum, self.zscore, self.expected_overlap, and self.real_overlap, which can be used for plotting purposes.
        Creates two signals, one corresponding to the cos and the other to the sin signal, as self.cos_signal and self.sin_signal
        In order to do a correlation without opening a file and populating self.child_row and self.parent_row, assign these variables manually.'''
        child_fft = scipy.fftpack.fft(self.child_row['data'])
        parent_fft = scipy.fftpack.fft(self.parent_row['data'])
        child_inverse_conjugate = -child_fft.conjugate()
        fft_auto_child = child_fft*child_inverse_conjugate
        fft_convolution = parent_fft*child_inverse_conjugate
        self.convolution_spectrum = numpy.abs(scipy.fftpack.ifft(fft_convolution/fft_auto_child)) #Roth window saved in self.convolution_spectrum
        assert self.time_inacc < sec/2, "This system clock cannot *possibly* be that inaccurate!"
        expected_tdiff = int((self.child_row['utcendtime']-self.parent_row['utcendtime'])*hz) #our initial guess for the location of the tdiff peak
        sample_inacc = int(self.time_inacc*hz) #around the guessed place of tdiff.
        if expected_tdiff-sample_inacc < 0: expected_tdiff = sample_inacc
        elif expected_tdiff+sample_inacc > sample_size: expected_tdiff = sample_size-sample_inacc
        cropped_convolution_spectrum = self.convolution_spectrum[expected_tdiff-sample_inacc:expected_tdiff+sample_inacc]
        #later measurements of tdiff will have a 0 point at expected_tdiff-sample_inacc and a total length of around 2*sample_inacc (may wrap around)
        tdiff = numpy.argmax(cropped_convolution_spectrum)
        tdiff += expected_tdiff-sample_inacc #offset for the real convolution_spectrum
        self.zscore = (self.convolution_spectrum[tdiff]-numpy.average(self.convolution_spectrum))/numpy.std(self.convolution_spectrum)
        #timespan = math.fabs(child_row['utcendtime'] - parent_row['utcendtime']) + sec
        self.expected_overlap = (float(sec) - 1.0*math.fabs(self.child_row['utcendtime']-self.parent_row['utcendtime']))*hz
        self.real_overlap = (float(sec) - 1.0*math.fabs(float(tdiff)/hz))*hz
        int_delay = int(numpy.copysign(numpy.floor(numpy.fabs(tdiff)), tdiff))
        self.abs_delay = int(abs(int_delay)) #always positive :)
        parent_signal, child_signal = self.parent_row['data'][self.abs_delay:], self.child_row['data'][0:len(self.child_row['data'])-self.abs_delay]
        h_child_length = len(child_signal)
        h_child_new_length = int(2**numpy.ceil(numpy.log2(h_child_length)))
        h_child_diff_length = h_child_new_length - h_child_length
        h_child_signal = scipy.fftpack.hilbert(numpy.append(child_signal, numpy.zeros(h_child_diff_length)))[0:h_child_length]
        self.cos_signal, self.sin_signal = evaluate("parent_signal*child_signal"), evaluate("h_child_signal*parent_signal")
    def fringe_rate_map(self, ra, dec):
        '''Calculates the coefficients for a fringe rate map with the UV Rate and Baseline length for a given timestamp using pyephem.
        If self.obs_site does not exist already, create it as an ephem.Observer() object and initialize it with lat and long.'''
        self.utcmidtime = (self.child_row['utcendtime']+self.parent_row['utcendtime']-self.integrationtime)/2.0 # "the middle" timestamp, reduces error from LST
        self.obs_site.date = datetime.datetime.utcfromtimestamp(self.utcmidtime)
        local_sidereal_time = self.obs_site.sidereal_time()
        x, y = self.detector_pos[self.child_row['id']][0]-self.detector_pos[self.parent_row['id']][0], self.detector_pos[self.child_row['id']][1]-self.detector_pos[self.parent_row['id']][1]
        hour_angle = local_sidereal_time - ephem.hours(str(ra)) #ra should be in hh:mm:ss
        decl = ephem.degrees(str(dec)) #dec should also be in dd:mm:ss
        deriv_ha = 2.0*numpy.pi/self.secs_in_sidereal_day
        self.du = x*numpy.cos(hour_angle)*deriv_ha-y*numpy.sin(hour_angle)*deriv_ha
        self.dv = x*numpy.sin(decl)*numpy.sin(hour_angle)*deriv_ha+y*numpy.sin(decl)*numpy.cos(hour_angle)*deriv_ha
        self.complex_visibility = self.cos_signal + 1.0j*self.sin_signal
        self.complex_visibility = numpy.append(self.complex_visibility,
                                               numpy.zeros(2**(numpy.ceil(numpy.log2(len(self.complex_visibility))))-len(self.complex_visibility))) #zero-padding for efficiency (2**n)
        self.fringe_rate_spectrum = scipy.fftpack.fft(self.complex_visibility)
        self.omega = numpy.argmax(self.fringe_rate_spectrum)
        self.xcoeff = 2.0*numpy.pi*self.du
        self.ycoeff = 2.0*numpy.pi*self.dv
    def save_correlation_spectrum(self):
        return self.convolution_spectrum
    def save_complex_vis(self):
        return self.complex_visibility
    def save_fringe_rate_spectrum(self):
        return self.fringe_rate_spectrum
    def save_frm(self):
        fringe_fields = ('id','omega','xcoeff','ycoeff','zscore','utcmidtime')
        return (fringe_fields,
                ((self.parent_row['id'], self.child_row['id']), self.omega, self.xcoeff, self.ycoeff, self.zscore, self.utcmidtime))
    def save_data(self):
        '''Could be extended to save any of the attributes: the correlation spectrum, the complex visibility, and the fringe-rate map.
        Those attributes are made available already.
        If save_data wants to use more than the basic save_frm() data, a different pytables class should be passed to initiate_hdf5.'''
        row = self.save_table.row
        row_data = self.save_frm()
        #assert row.nrows == len(row_data)
        for n in range(len(row_data[0])):
            row[row_data[0][n]] = row_data[1][n]
        row.append()
        self.save_table.flush()

def correlate(filename, quality=0.5): #quality denotes the minimum permissible overlap between two samples
    assert quality <= 1.0+1e-21, "The quality of crop cannot be over 100%."
    c = correlator()
    c.read_hdf5(filename)
    c.initiate_hdf5(fringe)
    ra, dec = (raw_input("Right Ascension of the source (hh:mm:ss): "), raw_input("Declination of the source (dd:mm:ss): "))
    rower = c.next_row()
    total_rows = c.h5file.root.detector.readout.nrows
    initial_time = time.time()
    for indicator in rower:
        cycle_time = time.time()
        c.correlate()
        correlate_time = time.time()
        #print "  %s seconds for correlate." % (correlate_time-filter_time)
        c.fringe_rate_map(ra, dec)
        frm_time = time.time()
        #print "  %s seconds for frm." % (frm_time-correlate_time)
        c.save_data()
        save_time = time.time()
        total_time = save_time-cycle_time
        print "Correlated (%s,%s) out of %s for baseline (%s,%s) in %.2f secs. %.2f%% corr, %.2f%% frm" % (indicator[0][0],
                                                                                                                        indicator[0][1],
                                                                                                                        total_rows,
                                                                                                                        indicator[1][0],
                                                                                                                        indicator[1][1],
                                                                                                                        total_time,
                                                                                                                        100.0*(correlate_time-cycle_time)/total_time,
                                                                                                                        100.0*(frm_time-correlate_time)/total_time)
    print "Correlation done in %s seconds." % (time.time()-initial_time)
        
#rtl.correlate('2012-10-22-6h11m_0.hd5')

