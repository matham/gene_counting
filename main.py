import csv
from functools import partial
from itertools import product
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import gmean
from os.path import join, exists
from os import mkdir
from scipy import stats
import matplotlib
import eleven
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pandas as pd

font = {'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)


first_data_col = 'ACT1_a'

gene = [
    'ARC', 'BDNF1', 'CREB1', 'EGR1', 'ERK1', 'ERK2', 'FOS', 'PRKR2A', 'PRKR2B']
gene_hk = ['ACT', 'GAPDH', 'PGK1']
efficiency = {
    "ACT": 0.929393051500794,
    "ARC": 0.9954001613524,
    "BDNF1": 0.924138179754939,
    "CREB1": 0.863653904237571,
    "EGR1": 0.960021930027294,
    "ERK1": 1.00796006493764,
    "ERK2": 0.856497692837879,
    "FOS": 0.963037975178076,
    "GAPDH": 0.9954001613524,
    "PRKR2A": 0.994738836814715,
    "PRKR2B": 0.935255682067943,
    "PGK1": 1.03198124224608
}

region = ['CER', 'CTX', 'HPC', 'OB', 'STR', 'Yellow']
day = ['1', '2', '4', '6']
delay = ['pre', '0', '15', '30', '60']


class GeneProduct(object):
    '''Houses data for a particular gene, region, day, delay combo.
    `reps` are the 1-3 rep ct values. `dct` is our computed 1-3 dct values.
    Similarly for `ddct`, and `amount`.
    '''

    __slots__ = ('name', 'day', 'delay', 'region', 'row', 'gene', 'reps',
                 'avg', 'stdev', 'ng', 'dct', 'ddct', 'amount', 'plate')

    def __init__(self, *largs, **kwargs):
        seen = set()
        for name, val in zip(self.__slots__, largs):
            setattr(self, name, val)
            seen.add(name)
        for name, val in kwargs.items():
            setattr(self, name, val)
            seen.add(name)
        for name in self.__slots__:
            if name not in seen:
                setattr(self, name, None)

    def clear_computed_fields(self):
        self.dct = self.ddct = self.amount = None


def pad_to(list_like, pad, val=''):
    '''Pads the list like object into a new list with a minimum of
    `pad` elements. The padded elements have value `val`.

    Returns the new list.
    '''
    if list_like is not None:
        res = list(list_like)
        if len(res) >= pad:
            return res
        return res + [val] * max(0, pad - len(res))
    return [val] * pad


def getattr_none(obj, name):
    '''Similar to `getattr`, except if `obj` is None it returns None rather
    than raising an exception.
    '''
    if name is None:
        return None
    return getattr(obj, name)


def get_values(data, attr, sort_order):
    '''Find all the possible values that attr `attr` of all the GeneProducts'
    items can take and return a list of only those found in `sort_order`
    and in the same order as they occur in `sort_order`.

    `data` is a list of GeneProduct instances.
    `attr` is the attr name, e.g. `dct`.
    `sort_order` is the list of possible values to extract.

    Returns the sorted values.
    '''
    items = list(set([getattr(item, attr) for item in data]))
    vals = []
    for val in sort_order:
        if val in items:
            vals.append(val)
            items.remove(val)

    # assert not items
    return vals


def remove_outliers(data):
    '''Removes from data the bad GeneProduct items that are pre-specified
    outliers.
    Returns a new filtered data list.
    '''
    res = []

    for item in data:
        if item.plate == '2' and item.region in ('CTX', 'STR') and item.gene in ('FOS', 'PRKR2A'):
            continue
        res.append(item)
    return res


def read_bahvior_data(filename):
    '''Reads the csv file that contains the result of the behavior data.
    The first three columns are the day number (1-6), the sac day, and the
    mouse name. The remaining columns are in (1-6) multiples of 20 where each
    20 is the result of the 20 trials of that day.
    The first row is ignored.

    Returns a dict whose keys are mouse names. Values are a 6-elem list, where
    each element is either None or a 20 element list with the behavior result
    of that day.
    '''
    with open(filename, 'rb') as fh:
        rows = list(csv.reader(fh))

    data = {}
    for row in rows[1:]:
        day, name, vals = row[0], row[2].lower(), row[3:23]
        try:
            for i, val in enumerate(vals):
                vals[i] = int(val) if val.lower() != 'x' else 'X'
        except:
            print('Skipping day {}: {}'.format(day, name))
            continue

        if name not in data:
            data[name] = [None, ] * 6

        animal_data = data[name]
        animal_data[int(day) - 1] = vals

    return data


def read_data(filename):
    '''Returns a list of `GeneProduct` containing the gene data.
    '''
    with open(filename, 'rb') as fh:
        rows = list(csv.reader(fh))

    header = rows[0]
    data_col = header.index(first_data_col)
    day_col, delay_col, region_col, name_col, plate_col = \
        map(lambda x: header.index(x),
            ['Train', 'Time', 'Region', 'Name', 'Plate'])
    data = []

    for i, row in enumerate(rows[1:]):
        # get the day/delay/region for this row
        day, delay, region = row[day_col], row[delay_col], row[region_col]
        mouse_id, plate_id = row[name_col], row[plate_col]

        # now create a data tuple for every gene using this row
        for g in gene + gene_hk:
            ng = stdev = avg = None
            reps = []

            for name, val in zip(header[data_col:], row[data_col:]):
                if not name.lower().startswith(g.lower()):
                    continue

                # found a col that matches this gene, find the type
                selector = name.split('_')[1].lower()
                if selector == 'ng':
                    ng = val
                elif selector == 'avg':
                    avg = val
                elif selector == 'stdev':
                    stdev = val
                else:
                    reps.append(val)

            try:
                reps = map(float, [r for r in reps if r])
                reps = np.array(reps)
                item = GeneProduct(
                    mouse_id, day, delay, region, i, g, reps,
                    float(avg), float(stdev), float(ng), plate=plate_id
                )
            except ValueError:
                print('Skipping "{}" from row "{}"'.format(
                    (day, delay, region, g), row))
                continue
            data.append(item)
    return data


def filter_data(data, day=day, region=region, delay=delay, gene=gene):
    '''Filters the GeneProduct elements in `data` and removes any elements
    whose day, region, delay, or gene values our not in the day, region, delay,
    and gene parameter values.

    Returns a new filtered list.

    .. warning:
        The default values of day, region, delay, and gene are taken from their
        global values.
    '''
    return [d for d in data if d.day in day and d.delay in delay and
            d.region in region and d.gene in gene]


def dump_data(data, filename, behavior_data={}):
    '''Dumps the data read by `read_data` (`data`) and `read_bahvior_data`
    (`behavior_data`) into the file `filename`.
    '''
    header = ['D{}T{}'.format(d + 1, t + 1) for d in range(6) for t in range(20)]
    header = ','.join(header)
    with open(filename, 'wb') as fh:
        fh.write(
            'Name,Plate,Day,Delay,Region,Gene,ct1,ct2,ct3,dct1,dct2,dct3,'
            'ddct1,ddct2,ddct3,amount1,amount2,amount3,{}\n'.format(header))
        for item in data:
            vals = [item.name, item.plate, item.day, item.delay, item.region,
                    item.gene]
            vals.extend(pad_to(item.reps, 3))
            vals.extend(pad_to(item.dct, 3))
            vals.extend(pad_to(item.ddct, 3))
            vals.extend(pad_to(item.amount, 3))

            days = behavior_data.get(item.name.split(' ', 1)[0].lower())
            if days:
                for day in days:
                    if day is not None:
                        assert len(day) == 20
                        vals.extend(day)
                    else:
                        vals.extend(['', ] * 20)

            fh.write(','.join(map(str, vals)))
            fh.write('\n')


def format_csv(data):
    '''Returns a string buffer representation of the GeneProduct data in
    `data`. Returns a 3 column string with the row, gene and rep value for
    each rep.
    '''
    output = StringIO()
    output.write('Sample,Target,Cq\n')
    for item in data:
        for val in item.reps:
            output.write('r{},{},{}\n'.format(item.row, item.gene, val))
    output.seek(0)
    return output


def select_housekeepr(csv_buffer, sample_value):
    '''Ranks the housekeeping genes. `csv_buffer` is the string buffer
    returned by `format_csv`. `sample_value` is a value of first column of that
    data. It could be any value from that column, except from the header row.
    Required by the algo.

    Returns a string indicating the ranking.
    '''
    df = pd.read_csv(csv_buffer, sep=',')
    censored = eleven.censor_background(df)
    ranked = eleven.rank_targets(censored, gene_hk, sample_value)
    return ranked


def compute_dct_exponentiated(data, housekeeper, mean_func=np.mean, pre_avg=False):
    # find the mean of the housekeeping genes
    norms = defaultdict(list)
    for item in data:
        if item.gene not in housekeepers:
            continue

        norms[item.row].append(item)

    for row, items in norms.items():
        items = [np.power(1 + efficiency[item.gene], np.mean(item.reps)) for item in items]
        norms[row] = mean_func(items)

    for item in data:
        if pre_avg:
            item.dct = [norms[item.row] / np.power(1 + efficiency[item.gene], np.mean(item.reps))]
        else:
            item.dct = norms[item.row] / np.power(1 + efficiency[item.gene], item.reps)


def set_amount_from_dct_exponentiated(data):
    for item in data:
        item.amount = item.dct


def compute_ddct_exponentiated(
        data, norm_keys=('delay', 'day'), norm_vals=('pre', '1'),
        norm_hash=('gene', 'region'), mean_func=np.mean):
    pool = defaultdict(list)
    for item in data:
        if not all([getattr(item, k) == v for k, v in zip(norm_keys, norm_vals)]):
            continue
        key = tuple([getattr(item, k) for k in norm_hash])
        pool[key].append(item)

    for item in data:
        key = tuple([getattr(item, k) for k in norm_hash])
        items = pool[key]


        item.ddct = item.dct/pool[key]


def compute_dct(data, housekeepers):
    '''Computes and populates the `dct` attr of the GeneProduct items in
    `data` using the housekeepers.

    `housekeepers` is a list of the names of the housekeepers to use to
    normalize.
    '''
    norms = defaultdict(list)
    for item in data:
        if item.gene not in housekeepers:
            continue

        norms[item.row].append(np.mean(item.reps))

    for row, ct_vals in norms.items():
        norms[row] = np.mean(ct_vals)

    for item in data:
        item.dct = item.reps - norms[item.row]


def compute_amount_from_dct(data):
    '''Computes and populates the `amount` attr of the GeneProduct items in
    `data` using the `dct` data.

    When used, the amount will be relative to the housekeepers and will not
    be the fold change as computed when using ddct.
    '''
    for item in data:
        item.amount = np.power(1 + efficiency[item.gene], -item.dct)


def compute_ddct(data, norm_keys=('delay', 'day'), norm_vals=('pre', '1'),
                              norm_hash=('gene', 'region')):
    '''Computes and populates the `ddct` attr of the GeneProduct items in
    `data` using the dct values matching the norm_keys/vals.

    To normalize dct into ddct we need reference dct data. The `norm_hash` is
    the attr names of GeneProduct items used to partition the data. I.e. the
    default of gene and region, we partition the data into all possible
    combinations of gene and region.

    Then, for each combination, we first get all the data in that set. Then,
    we find a subset which we average and subtract from the dct values in order
    to get the ddct values.

    This subset is found as follows. For each combination, get the items
    whose attrs have names found in `norm_keys` and whose corresponding values
    are the respective values in `norm_vals`. All those that match are combined
    into a set and then averaged.
    '''
    pool = defaultdict(list)
    for item in data:
        if not all([getattr(item, k) == v for k, v in zip(norm_keys, norm_vals)]):
            continue
        key = tuple([getattr(item, k) for k in norm_hash])

        pool[key].append(np.mean(item.dct))

    for key, items in pool.items():
        pool[key] = np.mean(items)

    for item in data:
        key = tuple([getattr(item, k) for k in norm_hash])
        if key not in pool:
            continue
        item.ddct = item.dct - pool[key]


def compute_amount_from_ddct(data):
    '''Computes and populates the `amount` attr of the GeneProduct items in
    `data` using the `ddct` data.

    When used, the amount will be relative to the housekeepers AND
    relative to the reference day.
    '''
    for item in data:
        if item.ddct is None:
            continue
        item.amount = np.power(1 + efficiency[item.gene], -item.ddct)


def get_unique_dir_name(ddct, scatter, x3, genes, gene_i, exp_domain, mean_func,
                        filter_outliers, gene_housek, **kwargs):
    '''Returns a unique directory name used to store the results given the
    values of the parameters.

    `ddct` is either 'dct' or 'ddct'.
    `scatter` is 'scatter' or 'error_bars'.
    `x3` is one of 'day', 'gene', 'region'.
    `genes` is the list of genes shown in the plot.
    `gene_i` is the number of the gene group when we split the genes into 3
        groups.
    `exp_domain` is unused.
    `mean_func` is unused.
    `filter_outliers` is True or False indicating whether we should filter
        outliers.
    `gene_housek` is the list of housekeeping genes to use.
    '''
    name = '{}, {}, {}'.format(ddct, scatter, x3)
    if len(genes) > 1:
        name = '{}, g{}'.format(name, gene_i)
    if exp_domain:
        name = '{} - exponentiated'.format(name)
        if mean_func is gmean:
            name = '{}, gmean'.format(name)
    if filter_outliers:
        name = '{}, no_outliers'.format(name)
    name = '{}, hk={}'.format(name, ','.join(gene_housek))
    return name

def plot_data(
        data, output, x1='delay', x2=None, x3=None, show_bars=True,
        avg_reps=False, show_error=True, ylabel=''):
    keys = 'gene', 'region', 'day', 'delay'
    selection = {k: get_values(data, k, globals()[k]) for k in keys}
    x1_labels = selection.pop(x1)
    x2_labels = selection.pop(x2, None)
    x3_labels = selection.pop(x3, None)
    variables = selection.keys()  # the variables which are constant for a plot

    # find all the combos of the variables we specify beforehand for the plot
    for spec in product(*[selection[var] for var in variables]):
        sorted_data = defaultdict(list)

        # first filter the data to only those selected by spec
        curr_data = [
            d for d in data
            if all([getattr(d, k) == val for k, val in zip(variables, spec)])
        ]

        for item in curr_data:
            sorted_data[(getattr_none(item, x1), getattr_none(item, x2), getattr_none(item, x3))].append(item)

        with open(join(output, 'binned_data.csv'), 'ab') as fh:
            for keys, vals in sorted_data.items():
                if avg_reps:
                    items = [np.mean(item.amount) for item in vals if item.amount is not None]
                else:
                    items = [v for item in vals for v in
                             (item.amount if item.amount is not None else [])]
                fh.write(','.join(list(spec) + [k for k in keys if k is not None] + map(str, items)))
                fh.write('\n')

        # now, we have the data sorted by x1/x2 and we just need to do the plot
        title = []
        for v, n in sorted(list(zip(variables, spec)), key=lambda x: x[0]):
            title.append('{}; {}'.format(v, n))
        title = ', '.join(title)

        fig, axs = plt.subplots(
            figsize=(18, 10), sharey=True, ncols=1 if x3_labels is None else len(x3_labels))
        try:
            if x2 is None:
                show_plot1(axs, sorted_data, title, x1, x1_labels,
                           show_bars=show_bars, avg_reps=avg_reps, ylabel=ylabel)
            else:
                if x3_labels is None:
                    show_plot2(axs, sorted_data, title, x1, x1_labels, x2_labels,
                               show_bars=show_bars, avg_reps=avg_reps, ylabel=ylabel)
                else:
                    max_val, min_val = 0, 1000
                    for idx, (label3, ax) in enumerate(zip(x3_labels, axs)):
                        filt_data = {k: v for k, v in sorted_data.items() if k[2] == label3}
                        max_val2, min_val2 = show_plot2(
                            ax, filt_data,
                            '{}, {}; {}'.format(title, x3, label3), x1, x1_labels, x2_labels,
                            show_bars=show_bars, avg_reps=avg_reps, show_ylabel=not idx,
                            show_legend=idx == len(axs) - 1, show_error=show_error, ylabel=ylabel)

                        max_val, min_val = max(max_val, max_val2), min(min_val, min_val2)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.xaxis.set_ticks_position('bottom')
                        if idx:
                            ax.spines['left'].set_visible(False)
                            ax.yaxis.set_ticks_position('none')
                        else:
                            ax.yaxis.set_ticks_position('left')

                    sp = (max_val - min_val) * .2
                    for ax in axs:
                        ax.set_ylim([min_val - sp, max_val + sp])

        except Exception as e:
            print e
            raise
        plt.tight_layout()
        fig.savefig(join(output, '{}.pdf'.format(title)))
        plt.close()


def show_plot1(ax, data, title, x1label, x1_labels, show_bars, avg_reps, ylabel=''):
    res = np.zeros(len(x1_labels))
    err = np.zeros(len(x1_labels))
    data = {k[0]: v for k, v in sorted_data.items()}

    for i, label in enumerate(x1_labels):
        if label not in data:
            raise Exception('Cannot find "{}"'.format(label))

        curr_items = data[label]
        items = [v for item in curr_items for v in item.amount]
        assert curr_items
        assert items
        res[i] = np.mean(items)
        err[i] = stats.sem(items)

    rects1 = ax.bar(np.arange(len(x1_labels)), res, yerr=err, align='center')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x1label.capitalize())
    ax.set_xticklabels(x1_labels)
    ax.set_xticks(np.arange(len(x1_labels)))
    ax.legend((rects1[0], ), (x1label, ))
    ax.set_title(title)


def show_plot2(ax, data, title, x1label, x1_labels, x2_labels, show_bars, avg_reps,
               show_ylabel=True, show_legend=True, show_error=True, ylabel=''):
    rects = [None, ] * len(x2_labels)
    width = (1 - .2) / float(len(x2_labels))
    cm = plt.get_cmap('brg')
    colors = [cm(1. * i / len(x2_labels)) for i in range(len(x2_labels))]

    data2 = defaultdict(dict)
    for (k1, k2, k3), values in data.items():
        data2[k2][k1] = values
    data = data2
    max_val, min_val = 0, 1000

    for i, (label2, color) in enumerate(zip(x2_labels, colors)):
        if label2 not in data:
            print('Cannot find "{}" for {}'.format(label2, title))
            continue

        res = np.zeros(len(x1_labels))
        err = np.zeros(len(x1_labels))
        x1_data = data[label2]

        x = np.arange(len(x1_labels))
        xp = x + i * width
        xdata = []
        ydata = []
        for l, label in enumerate(x1_labels):
            if label not in x1_data:
                print('Cannot find "{}"'.format(label))
                continue

            curr_items = x1_data[label]
            if avg_reps:
                items = [np.mean(item.amount) for item in curr_items if item.amount is not None]
            else:
                items = [v for item in curr_items for v in
                         (item.amount if item.amount is not None else [])]
            if not items:
                continue
            ydata.extend(items)
            xdata.extend([x[l], ] * len(items))
            assert curr_items
            assert items
            res[l] = np.mean(items)
            err[l] = stats.sem(items)
            dot_colors = [cm(1. * j / len(curr_items)) for j in range(len(curr_items))]
#             for c, item in zip(dot_colors, curr_items):
#                 ax.scatter([xp[l] + width / 2.] * len(item.amount), item.amount, marker='.', color=c, s=40)

        err[np.isnan(err)] = 0
        if show_error:
            max_val, min_val = max(np.max(res + err), max_val), min(np.min(res - err), min_val)
        else:
            max_val, min_val = max(np.max(res), max_val), min(np.min(res), min_val)

        if show_bars:
            rects[i] = ax.bar(xp, res, width, color=color, ecolor='k', yerr=err, label=label2)
            for l, val in enumerate(res):
                ax.plot([l + i * width, l + (i + 1) * width], [val, val], color='k')
        elif show_error:
            ax.errorbar(x, res, color=color, ecolor=color, yerr=err, label=label2)
            ax.plot(x, res, '.-', color=color, markersize=20)
        else:
            ax.plot(x, res, '-', color=color, label=label2)
            ax.scatter(xdata, ydata, marker='.', color=color, s=60)

    if show_ylabel:
        ax.set_ylabel(ylabel, **font)
    ax.set_xlabel(x1label.capitalize(), **font)
    if show_bars:
        ax.set_xticks(np.arange(len(x1_labels)) + 2 * width)
        if show_labels:
            ax.legend([rect[0] for rect in rects], x2_labels, **font)
    else:
        ax.set_xticks(x)
        ax.set_xlim([-.25, len(x) - .75])
        if show_legend:
            ax.legend()
    ax.set_xticklabels(x1_labels)
    ax.set_title(title)
    return max_val, min_val


if __name__ == '__main__':
    filename = r'C:\Users\Matthew Einhorn\Desktop\20150903a_Analysis_JP.csv'
    behavior_filename = r'C:\Users\Matthew Einhorn\Desktop\Michelle\behavior data.csv'
    output = r'C:\Users\Matthew Einhorn\Desktop\Michelle\figures'

    # read and filter the data.
    original_data = read_data(filename)
    original_data = filter_data(original_data, gene=gene_hk + gene, region=region[:-1])
    behavior_data = read_bahvior_data(behavior_filename)

    for ddct in ('ddct', 'dct'):
        for exp_domain in (False, ):
            for mean_func in (np.mean, gmean) if exp_domain else (None, ):
                for scatter in ('scatter', 'error_bars', ):
                    for x3 in ('day', 'gene', 'region'):
                        for filter_outliers in (True, False):
                            for gene_housek in (gene_hk[:1], ):
                                if x3 == 'gene':
                                    genes = [['ARC', 'EGR1', 'FOS'],
                                             ['BDNF1', 'ERK1', 'ERK2'],
                                             ['CREB1', 'PRKR2A', 'PRKR2B']]
                                else:
                                    genes = [['ARC', 'BDNF1', 'CREB1', 'EGR1', 'ERK1', 'ERK2', 'FOS', 'PRKR2A', 'PRKR2B']]
                                for gene_i, gene in enumerate(genes):
                                    name = get_unique_dir_name(**locals())
                                    cur_output = join(output, name)
                                    if not exists(cur_output):
                                        mkdir(cur_output)

                                    data = original_data[:]
                                    for item in data:  # clear previous vals
                                        item.clear_computed_fields()

                                    if filter_outliers:
                                        data = remove_outliers(data)

                                    buff = format_csv(data)
                                    print(select_housekeepr(buff, 'r{}'.format(data[0].row)))

                                    if not exp_domain:
                                        compute_dct(data, housekeepers=gene_housek)
                                        if ddct == 'dct':
                                            compute_amount_from_dct(data)
                                        else:
                                            compute_ddct(data)
                                            compute_amount_from_ddct(data)
                                    else:
                                        pass

                                    data = filter_data(data)
                                    dump_data(data, join(cur_output, 'processed_data.csv'), behavior_data=behavior_data)
                                    ylabel = 'Amount' if ddct == 'dct' else 'Fold change from Day1 pre'

                                    plot_data(data, cur_output, x2='region' if x3 != 'region' else 'day', x1='delay', x3=x3,
                                              show_bars=False, avg_reps=True, show_error=scatter == 'error_bars', ylabel=ylabel)
