from utils.test_utils import read_config_file


def get_figsize(columnwidth=241.14749, wf=1.0, hf=(5. ** 0.5 - 1.0) / 2.0, b_fixed_height=False):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex (pt). Get this from LaTeX
                             using \showthe\columnwidth (or \the\columnwidth)
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    if b_fixed_height:
        fig_height = hf
    print("fig_width", fig_width, "fig_height", fig_height)
    return [fig_width, fig_height]


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately
    turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val / 1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val / 1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]

    return new_tick_format


def get_function_value_offset(test_folder):
    """
    This function is for backward compatibility.
    Before 2021-07-02 the data saved during experiments contained the function value before all the nodes' sliding
    windows were full. Therefore data offset should be sliding_window_size in order to ignore this data.
    After 2021-07-02 the data starts only after all sliding windows are full, therefore offset should be 0.
    :param test_folder:
    :return:
    """
    date = test_folder.split("results_")[1].split("/")[0].split("_")[-2].split("-")
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    if (year > 2021) or (year == 2021 and month > 7) or (year == 2021 and month == 7 and day > 1):
        return 0
    conf = read_config_file(test_folder)
    sliding_window_size = conf["sliding_window_size"]
    return sliding_window_size
