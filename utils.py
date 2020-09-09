'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import math 

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
from tqdm import tqdm


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import torch.nn as nn
import torch
import numpy as np

def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=True):
    assert type(input_res) is tuple
    assert len(input_res) == 2
    batch = torch.FloatTensor(1, 3, *input_res)
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    out = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    params_count = get_model_parameters_number(flops_model)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count

def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'

def params_to_string(params_num):
    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + ' M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + ' k'

def print_model_with_flops(model, units='GMac', precision=3):
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model)
    model.apply(del_extra_repr)

def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num

def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask
    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                           torch.nn.LeakyReLU, torch.nn.ReLU6, torch.nn.Linear, \
                           torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.BatchNorm2d, \
                           torch.nn.Upsample, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
        return True

    return False


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += output_elements_count


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += active_elements_count


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    module.__flops__ += batch_size * input.shape[1] * output.shape[1]


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += np.prod(input.shape)

def bn_flops_counter_hook(module, input, output):
    module.affine
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += batch_flops

def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * output_height * output_width

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    batch_size = input.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, torch.nn.Conv2d):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                                 torch.nn.LeakyReLU, torch.nn.ReLU6)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (torch.nn.AvgPool2d, torch.nn.MaxPool2d, nn.AdaptiveMaxPool2d, \
                                 nn.AdaptiveAvgPool2d)):
            handle = module.register_forward_hook(pool_flops_counter_hook)
        elif isinstance(module, torch.nn.BatchNorm2d):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        elif isinstance(module, torch.nn.Upsample):
            handle = module.register_forward_hook(upsample_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None

neg_inf = -1 * math.pow(2,32)

def sram_traffic(
        dimension_rows=4,
        dimension_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
):


    # Dimensions of output feature map channel
    E_h = (ifmap_h - filt_h + strides) / strides
    E_w = (ifmap_w - filt_w + strides) / strides
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    num_h_fold = math.ceil(e2/dimension_rows)
    num_v_fold = math.ceil(num_filt/dimension_cols)

    cycles = 0

    read_cycles, util = gen_read_trace(
                            cycle = cycles,
                            dim_rows = dimension_rows,
                            dim_cols = dimension_cols,
                            num_v_fold = int(num_v_fold),
                            num_h_fold = int(num_h_fold),
                            ifmap_h = ifmap_h, ifmap_w= ifmap_w,
                            filt_h= filt_h, filt_w= filt_w,
                            num_channels= num_channels, stride=strides,
                            ofmap_h= int(E_h), ofmap_w= int(E_w), num_filters = num_filt,
                            filt_base= filt_base, ifmap_base= ifmap_base,
                            sram_read_trace_file= sram_read_trace_file
                            )

    write_cycles = gen_write_trace(
                        cycle = cycles,
                        dim_rows = dimension_rows,
                        dim_cols = dimension_cols,
                        #num_v_fold = int(num_v_fold),
                        #num_h_fold = int(num_h_fold),
                        ofmap_h = int(E_h), ofmap_w = int(E_w),
                        num_filters = num_filt,
                        ofmap_base = ofmap_base,
                        conv_window_size = r2c,
                        sram_write_trace_file = sram_write_trace_file
                        )

    cycles = max(read_cycles, write_cycles)
    str_cycles = str(cycles)
    return str_cycles
# End of sram_traffic()

        
def gen_read_trace(
        cycle = 0,
        dim_rows = 4, 
        dim_cols = 4,
        num_v_fold = 1,
        num_h_fold = 1,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w =3,
        num_channels = 3, stride = 1,
        ofmap_h =5, ofmap_w = 5, num_filters = 8, 
        filt_base = 1000000, ifmap_base = 0,
        sram_read_trace_file = "sram_read.csv",
        #sram_write_trace_file = "sram_write.csv"
):
    # Layer specific variables
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    e2 = ofmap_h * ofmap_w
    #num_ofmap_px = e2 * num_filters
    
    # Tracking variables
    local_cycle     = 0
    #remaining_px    = e2           # Need tracking for individual v folds
    #remaining_px     = []
    remaining_filt  = num_filters
    ifmap_done      = False
    filt_done       = False
    row_base_addr   = []
    row_clk_offset  = []
    row_ofmap_idx   = []
    v_fold_row      = []
    col_base_addr   = []
    col_clk_offset  = []
    v_fold_col      = []
    h_fold_col      = []
    lane_done       = []
    v_fold_barrier  = []

    # Variables for utilization calculation
    rows_used = 0
    cols_used = 0
    util      = 0

    # This initialization assumes num_rows << num_ofmap_px
    # The assignment logic needs to be modified if that is not the case
    for r in range(dim_rows):
        base_row_id = math.floor(r / ofmap_w) * stride
        base_col_id = r % ofmap_w * stride
        base_addr  = base_row_id * hc + base_col_id * num_channels 

        if r < e2:
            clk_offset = r * -1             # Clock offset takes care of the skew due to store and forward
        else:
            clk_offset = neg_inf            # In case num_ofamp_px < dim_rows

        row_base_addr.append(base_addr)
        row_clk_offset.append(clk_offset)
        row_ofmap_idx.append(r)
        v_fold_row.append(0)
        v_fold_barrier.append(False)

    for c in range(dim_cols):
        base_addr = c * r2c

        # Anand: TODO
        if c < remaining_filt:
            clk_offset = c * -1
            lane_done.append(False)
        else:
            clk_offset = neg_inf
            lane_done.append(True)

        col_base_addr.append(base_addr)
        col_clk_offset.append(clk_offset)
        v_fold_col.append(0)
        h_fold_col.append(0)


    # Open tracefile for writing
    # outfile     = open(sram_read_trace_file, 'w')
    #ofmap_out   = open(sram_write_trace_file, 'w')

    # Adding progress bar
    tot  = e2 * num_v_fold
    #print("Total = " + str(tot))
    # pbar = tqdm(total=tot)

    # Generate traces here
    # The condition checks
    #       1)  if the all the input traces for last v fold is generated
    # and   2)  if all the filter traces have been generated
    #while(remaining_px[num_v_fold-1] > 0) or (filt_done == False):
    while(ifmap_done == False) or (filt_done == False):
        ifmap_read = ""
        filt_read  = ""
        rows_used = 0
        cols_used = 0
        
        # Generate address for ifmap
        for r in range(dim_rows):

            if(row_clk_offset[r] >= 0):     # Take care of the skew

                inc = row_clk_offset[r]

                addr_row_offset = math.floor(inc / rc) * ifmap_w * num_channels
                addr_col_offset = inc % rc
                ifmap_addr = row_base_addr[r] + addr_row_offset + addr_col_offset 
                ifmap_read += str(int(ifmap_addr)) + ", "
                rows_used += 1
            else:
                ifmap_read += ", "

            row_clk_offset[r] += 1

            if (row_clk_offset[r] > 0) and (row_clk_offset[r] % r2c == 0):   #Completed MAC for one OFMAP px
                
                row_ofmap_idx[r] += dim_rows
                ofmap_idx = row_ofmap_idx[r]

                # Update progress bar
                # pbar.update(1)

                if ofmap_idx < e2:
                    row_clk_offset[r] = 0

                    base_row_id = math.floor(ofmap_idx / ofmap_w) * stride
                    base_col_id = ofmap_idx % ofmap_w * stride
                    base_addr  = base_row_id * hc + base_col_id * num_channels
                    row_base_addr[r] = base_addr

                else:
                    v_fold_row[r] += 1
                    #pbar.update(e2)

                    if(v_fold_row[r] < num_v_fold):
                        row_ofmap_idx[r]  = r

                        base_row_id = math.floor(r / ofmap_w) * stride
                        base_col_id = r % ofmap_w * stride
                        base_addr  = base_row_id * hc + base_col_id * num_channels
                        row_base_addr[r]  = base_addr

                        # Stall this col from proceeding until all the rows reach the v_fold boundary
                        if (r != 0) and ((v_fold_row[r] > v_fold_row[r-1]) or (v_fold_barrier[r-1] == True)):
                            row_clk_offset[r] = neg_inf
                            v_fold_barrier[r] = True
                        else:
                            row_clk_offset[r] = 0

                    else:
                        row_clk_offset[r] = neg_inf

        # Get out of the barrier one by one
        # IMPORTANT: The barrier insertion and recovery is in separate loops to ensure that
        #            in a given clock cycle insertion for all rows strictly happen before the release.
        #            The flag ensures only one col is released per cycle
        # Since indx 0 never enters the barrier, this should work fine
        flag = False
        for r in range(dim_rows):
            if v_fold_barrier[r] and flag==False:
                if (v_fold_row[r] == v_fold_row[r-1]) and (v_fold_barrier[r-1] == False):
                    v_fold_barrier[r] = False
                    flag = True
                    row_clk_offset[r] = row_clk_offset[r-1] -1

        # Check if all input traces are done
        ifmap_done = True
        for r in range(dim_rows):
            if row_clk_offset[r] > 0:
                ifmap_done = False

        # Generate address for filters
        for c in range(dim_cols):
            if(col_clk_offset[c] >= 0):     # Take care of the skew
                inc = col_clk_offset[c]
                
                filt_addr = col_base_addr[c] + inc + filt_base 
                filt_read += str(filt_addr) + ", "
                cols_used += 1
            else:
                filt_read += ", "

            col_clk_offset[c] += 1

            if(col_clk_offset[c] > 0) and (col_clk_offset[c] % r2c == 0):

                # Get the v fold this col is working on and check the status of input trace generation
                #rem_px = remaining_px[v_fold_col[c]]

                #Tracking on the basis of h_folds
                h_fold_col[c] += 1

                # Anand: Check if all the input traces are generated for the given v fold
                if (h_fold_col[c] < num_h_fold):
                    col_clk_offset[c] = 0
                else:
                    v_fold_col[c] += 1
                    filt_id = v_fold_col[c] * dim_cols + c

                    # All filters might not be active in the last fold
                    # Adding the filter ID check to ensure only valid cols are active
                    if(v_fold_col[c] < num_v_fold) and (filt_id < num_filters):
                        col_clk_offset[c] = 0
                        h_fold_col[c] = 0

                        base = filt_id * r2c
                        col_base_addr[c] = base

                    else:
                        col_clk_offset[c] = neg_inf
                        lane_done[c] = True

        # Check if all filter traces are generated
        filt_done = True
        for c in range(dim_cols):
            if lane_done[c] == False:
                filt_done = False

                                                
        # Write to trace file
        global_cycle = cycle + local_cycle
        entry = str(global_cycle) + ", " + ifmap_read + filt_read + "\n"
        # outfile.write(entry)

        this_util = (rows_used * cols_used) / (dim_rows * dim_cols)
        util += this_util

        # Update tracking variableslocal_cycle
        local_cycle += 1

    # pbar.close()
    # outfile.close()
    #ofmap_out.close()

    util_perc = (util / local_cycle) * 100

    return (local_cycle + cycle, util_perc)
# End of gen_read_trace()


def gen_write_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        #num_v_fold = 1,
        #num_h_fold = 1,
        ofmap_h = 5, ofmap_w = 5,
        num_filters = 4,
        ofmap_base = 2000000,
        conv_window_size = 9,                      # The number of pixels in a convolution window
        sram_write_trace_file = "sram_write.csv"
):

    # Layer specific variables
    r2c = conv_window_size
    e2  = ofmap_h * ofmap_w

    # Tracking variables
    id_row = []             # List of OFMAP ID for each row
    id_col = []             # List of filter ID for each col
    base_addr_col =[]       # Starting address of each output channel
    remaining_px  = e2
    remaining_filt= num_filters
    active_row = min(dim_rows, e2)
    active_col = min(dim_cols, num_filters)
    local_cycle = 0
    sticky_flag = False     # This flag is in place to fix the OFMAP cycle shaving bug

    for r in range(active_row):
        id_row.append(r)

    for c in range(active_col):
        id_col.append(c)

        base_col = c
        base_addr_col.append(base_col)

    #Open the file for writing
    # outfile = open(sram_write_trace_file,"w")

    #This is the cycle when all the OFMAP elements in the first col become available
    local_cycle = r2c + active_col - 1

    while (remaining_px > 0) or (remaining_filt > 0):

        active_row = min(dim_rows, remaining_px)

        for r in range(active_row):
            local_px = id_row[r]
            remaining_px -= 1
            id_row[r] += active_row     # Taking care of horizontal fold

            ofmap_trace = ""
            for c in range(active_col):
                addr = ofmap_base + base_addr_col[c] + local_px * num_filters
                ofmap_trace += str(addr) + ", "

            # Write the generated traces to the file
            entry = str(local_cycle + r) + ", " + ofmap_trace + "\n"
            # outfile.write(entry)

        # Take care of the vertical fold
        if remaining_px == 0:
            remaining_filt -= active_col

            # In case of vertical fold we have to track when the output of (0,0) is generated
            # Shifting back local cycles to capture the last OFMAP generation in (0,0) for this fold
            last_fold_cycle   = local_cycle + active_row
            local_cycle -= (active_row + active_col - 1)
            sticky_flag = True

            # There are more OFMAP channels to go
            if remaining_filt > 0:
                remaining_px = e2
                last_active_col = active_col
                active_col = min(remaining_filt, dim_cols)

                # Reassign col base addresses
                for c in range(active_col):
                    base_addr_col[c] += last_active_col

                active_row = min(dim_rows, remaining_px)
                # Reassign row base addresses
                for r in range(active_row):
                    id_row[r] = r

                local_cycle += r2c + active_col
                if local_cycle < last_fold_cycle:
                    local_cycle = last_fold_cycle


            else:   # Restore the local cycle to return to the main function
                local_cycle = last_fold_cycle
                #local_cycle += (active_row + active_col)
                #sticky_flag = False

        else:   # If this is not a vertical fold then it is business as usual
            local_cycle += max(r2c, active_row)

    # outfile.close()

    #if sticky_flag:
    #    local_cycle += (active_row + active_col)
    #    sticky_flag = False

    return(local_cycle + cycle)
# End of gen_write_trace()

def gemmCycles(dimension_rows, dimension_cols, ifmap_h, ifmap_w, filt_h, filt_w,
            num_channels, strides, num_filt):
        H = ifmap_h
        W = ifmap_w
        C = num_channels
        M = num_filt
        R = filt_h
        S = filt_w
        Stride = strides
        arrX = dimension_rows
        arrY = dimension_cols

        E = (H - R + Stride)/Stride
        F = (W - S + Stride)/Stride
    
        ## Reduce to Mat mul of A x B and  B X C
        numInput = E * F
        numTime  = R * S * C
        numFilter= M

        cycles = 0
        cycles = (numInput//arrX) * (numFilter//arrY) * (numTime + arrX + arrY - 1)

        if numInput % arrX > 0:
            cycles = cycles + (numFilter//arrY) * (numTime + (numInput % arrX) + arrY - 1)
        if numFilter % arrY > 0:
            cycles = cycles + (numInput//arrX) * (numTime + arrX + (numFilter % arrY) - 1)
        if numInput % arrX > 0 and numFilter % arrY > 0:
            cycles = cycles + (numInput//arrX) * (numTime + (numInput % arrX) + (numFilter % arrY) - 1)
        return cycles

class ForwardHook:
    def __init__(self, arraySize, mode):
        self.time = 0
        self.pointwiseConv = 0
        self.depthwiseConv = 0
        self.otherConv = 0
        self.arraySize = arraySize
        if mode == 'analytical':
            self.latencyFn = gemmCycles
        else:
            self.latencyFn = sram_traffic
    def __call__(self, module, module_in, module_out):
        inT = module_in[0]
        inDim_h, inDim_w = (inT.shape[2], inT.shape[3])
        inC = module.in_channels
        outC = module.out_channels
        k_h, k_w = module.kernel_size
        s_h, s_w = module.stride
        p_h, p_w = module.padding
        g = module.groups
        inDim_h = inDim_h + 2*p_h
        inDim_w = inDim_w + 2*p_w
        if g == 1:
            t = self.latencyFn(dimension_rows=self.arraySize, dimension_cols=self.arraySize, 
                            ifmap_h=inDim_h, ifmap_w=inDim_w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=inC,strides=s_h, num_filt=outC)
            # print('Group=1 ', inDim_h, inDim_w, k_h, k_w, inC, outC, t)
            t = int(t)
            if k_h == 1 and k_w == 1:
                self.pointwiseConv += t
            else:
                self.otherConv += t
        else:
            if k_h == 1:
                num1Dconv = inDim_h * outC 
                numFolds = num1Dconv/self.arraySize
                oneFoldTime = self.arraySize + k_w
                num1DconvRow = inDim_h/self.arraySize
                time = (math.ceil(numFolds)/s_w)*(oneFoldTime*math.ceil(num1DconvRow))
                time = math.ceil(time)
                t = time
                self.depthwiseConv += t
            elif k_w ==1 :
                num1Dconv = inDim_w * outC
                numFolds = num1Dconv/self.arraySize
                oneFoldTime = self.arraySize + k_h
                num1DconvRow = inDim_w/self.arraySize
                time = (math.ceil(numFolds)/s_w)*(oneFoldTime*math.ceil(num1DconvRow))
                time = math.ceil(time)
                t = time
                self.depthwiseConv += t
            else:
                t = self.latencyFn(dimension_rows=self.arraySize, dimension_cols=self.arraySize, 
                            ifmap_h=inDim_h, ifmap_w=inDim_w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=1,strides=s_h, num_filt=1)
                t = int(t)
                t = t*outC
                self.depthwiseConv += t

            # print('Group > 1 ', inDim_h, inDim_w, k_h, k_w, inC, outC, t)

        self.time += t
    
    def clear(self):
        self.time = 0
        self.pointwiseConv = 0
        self.depthwiseConv = 0
        self.otherConv = 0

def getModelProp(model, x):
    flops, parameter = get_model_complexity_info(model, (x.shape[2], x.shape[3]), print_per_layer_stat=False, as_strings=False)
    return flops, parameter

def getModelLatency(model, x, mode='analytical', arraySize=8):    
    hookfn = ForwardHook(arraySize, mode)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
    model(x)
    latency = hookfn.time
    hookfn.clear()

    return latency

def getModelLatencyBreakdown(model, x, mode='analytical', arraySize=8):    
    hookfn = ForwardHook(arraySize, mode)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
    model(x)
    totalLatency = hookfn.time
    otherConvLatency = hookfn.otherConv
    pointConvLatency = hookfn.pointwiseConv
    depthConvLatency = hookfn.depthwiseConv
    hookfn.clear()
    return [totalLatency, otherConvLatency, pointConvLatency, depthConvLatency]
