'''
Created on Sep 19, 2012

@author: chuxpy

The complete go-to-module on RTL data input, committing, etc.

Data Model:
Sky signal -> Receivers
Receivers+RHTimestamp+AntennaID -> Array
'''

import rtlsdr, datetime, itertools, multiprocessing, tables, sys, time, os, math

sec, hz = 1.0, 2**20
sample_size = int(sec*hz)

#metadata; info = tables.Float32Col(shape=3) #essential data for interpretation and shifting of samples: (sample_rate[hz], length[sec])
class sample(tables.IsDescription):
    id = tables.Int32Col(pos=0) #antenna ID
    utcendtime = tables.Time64Col(pos=1) #accurate to milliseconds
    data = tables.Float64Col(shape=sample_size, pos=2) #the actual data from (utcendtime-sec) to (utcendtime) with a step of hz.

def process_receiver(rtl_queue, rtl_num):
    sdr = rtlsdr.RtlSdr(rtl_num)
    sdr.sample_rate = hz
    sdr.center_freq = 1.4208e9
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
        for receiver_num in self.rtl_list:
            self.rtl_queue[receiver_num] = multiprocessing.Queue()
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
        return table
    def row_write(self, table, rtl_num, data, timestamp):
        '''Writes a row of data to a HDF5 table/array object.'''
        row = table.row
        row['id'] = rtl_num
        row['utcendtime'] = timestamp
        row['data'] = data
        row.append()
        table.flush()

def start(length=0, time_limit=0): #stop when length or time limit reached.
    length = int(length)
    start_time = time.time()
    current_length = 0
    sources = [0,1,2]
    rc = receiver(sources)
    queue = rc.get_queue()
    table = rc.initiate_hdf5()
    try:
        while True:
            for (p, q) in queue.items():
                row = q.get() #no timeout
                rc.row_write(table = table, rtl_num = row['Receiver'], data = row['Data'], timestamp = row['Timestamp'])
                if length > 0:
                    current_length += 1
                    if current_length > length:
                        sys.exit('%s entries recorded to the HDF5 table. :) Halted because of a length limit.' % current_length)
                if time > 0:
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
        return h5file.root.detector.readout
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

def correlate_for_true_delay(child_row, parent_row):
    return tdiff

def resample(row, var_delay):
    return row

def correlate(filename, quality=0.5): #quality denotes the minimum permissible overlap between two samples
    assert quality <= 1.0+1e-21, "The quality of correlation cannot be over 100%."
    c = correlator()
    tbl = c.read_hdf5(filename)
    for num, row in enumerate(tbl.itersorted(tbl.cols.utcendtime)):
        min_overlap = quality*sec
        iterated = enumerate(tbl.itersorted(tbl.cols.utcendtime, start=num, stop=tbl.nrows), start=num)
        try:
            anum, arow = iterated.next()
            while True:
                if not arow['utcendtime'] > (row['utcendtime'] + min_overlap):
                    #do all processing and passing here!
                    child_data, parent_data = crop(arow, row)
                    child_utcendtime, parent_utcendtime = arow['utcendtime'], row['utcendtime']
                    child_id, parent_id = arow['id'], row['id']
                    child_data =
                    anum, arow = iterated.next()
                else:
                    raise GetMeOutofThisLoop
        except StopIteration: #set stop index pls
            anum += 1
        except GetMeOutofThisLoop: #the stop index does not need to be set
            pass
        all_row_indices = range(num, anum)
        print '%s: %s. %s' %(num, all_row_indices, row[1]) #this is not part of final

correlate('2012-9-30-22h19m_0.hd5')
