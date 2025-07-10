def hello(input1, input2=None):
    print("Hello World!")

def print_list_3dp(my_list):
    my_formatted_list = [ '%.3f' % elem for elem in  my_list]
    # https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places
    # https://stackoverflow.com/questions/44639357/print-python-list-without-quotation-marks-or-space-after-commas
    print("time series = ",  ', '.join(my_formatted_list))
def print_list_3dp_vertically(my_list):
    my_formatted_list = [ '%.3f' % elem for elem in  my_list]
    # https://stackoverflow.com/questions/5982206/how-to-print-a-linebreak-in-a-python-function
    print('\n'.join(my_formatted_list))

# https://stackoverflow.com/questions/37130146/is-it-possible-to-detect-the-number-of-return-values-of-a-function-in-python
def transform_original(seq, seq_prob=None, prefix=None, prefix_prob=None):
    # prob_seq is for dummy purpose to ensure all the transformations have the same input parameters
    # So I can run a function on the function list
    return seq, None

def transform_single_ss(seq):
    ts = [None] * len(seq)
    for i in range(len(seq)):
        if seq[i] == '(':
            ts[i] = 1
        elif seq[i] == '.':
            ts[i] = 0
        elif seq[i] == ')':
            ts[i] = -1
        else:
            raise ValueError('The sequence contains invalid characters')  
    return ts

def transform_single_no_domain(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Single value mapping without domain knowledge
    """
    ts = [None] * len(seq)
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1 # No probability is used

        if seq[i] == 'A':
            ts[i] = 2 * prob
        elif seq[i] == 'C':
            ts[i] = 1 * prob
        elif seq[i] == 'G':
            ts[i] = -1 * prob
        elif seq[i] == 'U':
            ts[i] = -2 * prob
        elif seq[i] == '_':
            ts[i] = 0
        else:
            raise ValueError('The sequence contains invalid characters')  
    return ts, None

def transform_single(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Single value mapping with domain knowledge
    """
    ts = [None] * len(seq)
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1 # No probability is used

        if seq[i] == 'A':
            ts[i] = 2 * prob
        elif seq[i] == 'G':
            ts[i] = 1 * prob
        elif seq[i] == 'C':
            ts[i] = -1 * prob
        elif seq[i] == 'U':
            ts[i] = -2 * prob
        elif seq[i] == '_':
            ts[i] = 0
        else:
            raise ValueError('The sequence contains invalid characters')  
    return ts, None

def transform_single_multi_diff(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Grouped variable-length channel mapping
    """
    ts_1 = [None] * len(seq)
    ts_2 = [None] * len(seq)
    j = 0
    k = 0
    # ts_1[j] = 0
    # ts_2[k] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1

        if seq[i] == 'A':
            ts_1[j] = 1 * prob
            j += 1
        elif seq[i] == 'U':
            ts_1[j] = -1 * prob
            j += 1
        elif seq[i] == 'G':
            ts_2[k] = 1 * prob
            k += 1
        elif seq[i] == 'C':
            ts_2[k] = -1 * prob
            k += 1
        elif seq[i] == '_':
            # Group this into A/U Group
            ts_1[j] = 0
            j += 1
            # pass
        else:
            raise ValueError('The sequence contains invalid characters')  
    return ts_1[0:j], ts_2[0:k]

def transform_single_multi_eq(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Grouped fixed-length channel mapping
    """
    ts_1 = [None] * len(seq)
    ts_2 = [None] * len(seq)
    # ts_1[0] = 0
    # ts_2[0] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1
        if seq[i] == 'A':
            ts_1[i] = 1 * prob
            ts_2[i] = 0
        elif seq[i] == 'U':
            ts_1[i] = -1 * prob
            ts_2[i] = 0
        elif seq[i] == 'G':
            ts_1[i] = 0
            ts_2[i] = 1 * prob
        elif seq[i] == 'C':
            ts_1[i] = 0
            ts_2[i] = -1 * prob
        elif seq[i] == '_':
            ts_1[i] = 0
            ts_2[i] = 0
        else:
            raise ValueError('The sequence contains invalid characters')  
    return ts_1, ts_2

def transform_cum(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Cumulative mapping
    """
    if prefix:
        res1, res2 = transform_cum(prefix, prefix_prob)
    ts = [None] * (len(seq)+1)
    ts[0] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1
        if seq[i] == 'A':
            ts[i+1] = ts[i] + 2 * prob
        elif seq[i] == 'G':
            ts[i+1] = ts[i] + 1 * prob
        elif seq[i] == 'C':
            ts[i+1] = ts[i] - 1 * prob
        elif seq[i] == 'U':
            ts[i+1] = ts[i] - 2 * prob
        elif seq[i] == '_':
            ts[i+1] = ts[i]
        else:
            raise ValueError('The sequence contains invalid characters')
    try:
        return ([x + res1[-1] for x in ts]), None
    except:
        return ts, None
    
def transform_cum_multi_diff(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Cumulative grouped variable-length channel mappingng
    """
    if prefix:
        res1, res2 = transform_cum_multi_diff(prefix, prefix_prob) 
    ts_1 = [None] * (len(seq)+1)
    ts_2= [None] * (len(seq)+1)
    j = 0
    k = 0
    ts_1[j] = 0
    ts_2[k] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1
        if seq[i] == 'A':
            ts_1[j+1] = ts_1[j] + 1 * prob
            j += 1
        elif seq[i] == 'U':
            ts_1[j+1] = ts_1[j] - 1 * prob
            j += 1
        elif seq[i] == 'G':
            ts_2[k+1] = ts_2[k] + 1 * prob
            k += 1
        elif seq[i] == 'C':
            ts_2[k+1] = ts_2[k] - 1 * prob
            k += 1
        elif seq[i] == '_':
            # Do nothing
            pass
        else:
            raise ValueError('The sequence contains invalid characters')  
    try:
        return ([x + res1[-1] for x in ts_1[0:j+1]]), ([x + res2[-1] for x in ts_2[0:k+1]])
    except:
        return ts_1[0:j+1], ts_2[0:k+1]

def transform_cum_multi_eq(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Cumulative grouped fixed-length channel mappingng
    """
    if prefix:
        res1, res2 = transform_cum_multi_eq(prefix, prefix_prob) 
    ts_1 = [None] * (len(seq)+1)
    ts_2= [None] * (len(seq)+1)
    ts_1[0] = 0
    ts_2[0] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1
        if seq[i] == 'A':
            ts_1[i+1] = ts_1[i] + 1 * prob
            ts_2[i+1] = ts_2[i]
        elif seq[i] == 'U':
            ts_1[i+1] = ts_1[i] - 1 * prob
            ts_2[i+1] = ts_2[i]
        elif seq[i] == 'G':
            ts_1[i+1] = ts_1[i]
            ts_2[i+1] = ts_2[i] + 1 * prob
        elif seq[i] == 'C':
            ts_1[i+1] = ts_1[i]
            ts_2[i+1] = ts_2[i] - 1 * prob
        elif seq[i] == '_':
            ts_1[i+1] = ts_1[i]
            ts_2[i+1] = ts_2[i]
        else:
            raise ValueError('The sequence contains invalid characters')  
    try:
        return ([x + res1[-1] for x in ts_1]), ([x + res2[-1] for x in ts_2])
    except:
        return ts_1, ts_2

def transform_cum_multi_diff_reverse(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Cumulative grouped variable-length channel mappingng
    """
    if prefix:
        res1, res2 = transform_cum_multi_diff_reverse(prefix, prefix_prob) 
    ts_1 = [None] * (len(seq)+1)
    ts_2= [None] * (len(seq)+1)
    j = 0
    k = 0
    ts_1[j] = 0
    ts_2[k] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1
        if seq[i] == 'G':
            ts_1[j+1] = ts_1[j] + 1 * prob
            j += 1
        elif seq[i] == 'U':
            ts_1[j+1] = ts_1[j] - 1 * prob
            j += 1
        elif seq[i] == 'A':
            ts_2[k+1] = ts_2[k] + 1 * prob
            k += 1
        elif seq[i] == 'C':
            ts_2[k+1] = ts_2[k] - 1 * prob
            k += 1
        elif seq[i] == '_':
            # Do nothing
            pass
        else:
            raise ValueError('The sequence contains invalid characters')  
    try:
        return ([x + res1[-1] for x in ts_1[0:j+1]]), ([x + res2[-1] for x in ts_2[0:k+1]])
    except:
        return ts_1[0:j+1], ts_2[0:k+1]

def transform_cum_multi_eq_reverse(seq, seq_prob=None, prefix=None, prefix_prob=None):
    """
    Cumulative grouped fixed-length channel mappingng
    """
    if prefix:
        res1, res2 = transform_cum_multi_eq_reverse(prefix, prefix_prob) 
    ts_1 = [None] * (len(seq)+1)
    ts_2= [None] * (len(seq)+1)
    ts_1[0] = 0
    ts_2[0] = 0
    for i in range(len(seq)):
        if seq_prob:
            prob = seq_prob[i]
        else:
            prob = 1
        if seq[i] == 'G':
            ts_1[i+1] = ts_1[i] + 1 * prob
            ts_2[i+1] = ts_2[i]
        elif seq[i] == 'U':
            ts_1[i+1] = ts_1[i] - 1 * prob
            ts_2[i+1] = ts_2[i]
        elif seq[i] == 'A':
            ts_1[i+1] = ts_1[i]
            ts_2[i+1] = ts_2[i] + 1 * prob
        elif seq[i] == 'C':
            ts_1[i+1] = ts_1[i]
            ts_2[i+1] = ts_2[i] - 1 * prob
        elif seq[i] == '_':
            ts_1[i+1] = ts_1[i]
            ts_2[i+1] = ts_2[i]
        else:
            raise ValueError('The sequence contains invalid characters')  
    try:
        return ([x + res1[-1] for x in ts_1]), ([x + res2[-1] for x in ts_2])
    except:
        return ts_1, ts_2

