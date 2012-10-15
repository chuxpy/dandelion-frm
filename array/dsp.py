'''
Created on Sep 15, 2012

@author: chuxpy

Digitally processing and correlating signals from the receivers.
'''

import rtlsdr, numpy, scipy.signal
import matplotlib.pyplot as plt

def b_filter(order, low, high, sample_rate, filter_type='band'):
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


#### matplotlib section

def plot(time_series, data_series, label):
    plt.plot(time_series, data_series, label=label)

def figures(data, filtered):
    plot(numpy.linspace(0, 1.0, 2 ** 20, 'bo'), data, 'Raw signal')
    plot(numpy.linspace(0, 1.0, 2 ** 20, 'bo'), filtered, '100k bandwidth')
    plt.legend()

#### delay determination

def pretend_delay():
    return (0.0, 241.712)

#### interpolation

def interpolate(delays, twodata):
    if delays[0] > delays[1]:
        twodata[0], twodata[1] = twodata[1], twodata[0]
        delays[0], delays[1] = delays[1], delays[0]
    diff_delay = delays[1]-delays[0] #should always be > 0
    floating_point = diff_delay % 1 #get the decimal.
    top, bot = twodata[0][int(numpy.floor(diff_delay)):], twodata[1][0:-int(numpy.floor(diff_delay))]
    #Now we have two possible solutions, one by interpolating top, and one by interpolating bot.
    #May be feasible to do both and average, but I'm going to interpolate bot in this following code.
    interpolate_point = 1 - floating_point
    inter_x = numpy.r_[interpolate_point:(len(bot)-1e-9)][0:len(top)-1] #make sure length is exactly right.
    newbot = scipy.signal.cspline1d_eval(scipy.signal.cspline1d(bot), inter_x, dx=1.0, x0=0.0)
    return (top[1:], newbot) #now they're perfectly aligned.
'''Next feasible step: split the complex and make sure it gets fixed.'''
