import csv
from functools import partial
from itertools import product
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy import stats
import eleven
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pandas as pd


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

    __slots__ = ('day', 'delay', 'region', 'row', 'gene', 'reps', 'avg',
                 'stdev', 'ng', 'dct', 'ddct', 'amount')

    def __init__(self, *largs):
        for name, val in zip(self.__slots__, largs):
            setattr(self, name, val)


def pad_to(list_like, pad, val=''):
    res = list(list_like)
    if len(res) >= pad:
        return res
    return res + [val] * (pad - len(res))


def get_values(data, attr, sort_order):
    items = list(set([getattr(item, attr) for item in data]))
    vals = []
    for val in sort_order:
        if val in items:
            vals.append(val)
            items.remove(val)

    assert not items
    return vals


def read_data(filename, ignore_rep_errors=True):
    '''Returns a list of `GeneProduct` containing the gene data.
    '''
    with open(filename, 'rb') as fh:
        rows = list(csv.reader(fh))

    header = rows[0]
    data_col = header.index(first_data_col)
    day_col, delay_col, region_col = \
        map(lambda x: header.index(x), ['Train', 'Time', 'Region'])
    data = []

    for i, row in enumerate(rows[1:]):
        # get the day/delay/region for this row
        day, delay, region = row[day_col], row[delay_col], row[region_col]

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
                reps = map(
                    float, [r for r in reps if r or not ignore_rep_errors])
                item = GeneProduct(
                    day, delay, region, i, g, np.array(reps),
                    float(avg), float(stdev), float(ng)
                )
            except ValueError:
                print('Skipping "{}" from row "{}"'.format(
                    (day, delay, region, g), row))
                continue
            data.append(item)
    return data


def filter_data(data, day=day, region=region, delay=delay, gene=gene):
    return [d for d in data if d.day in day and d.delay in delay and
            d.region in region and d.gene in gene]


def dump_data(data, filename):
    with open(filename, 'wb') as fh:
        fh.write(
            'Day,Delay,Region,Gene,ct1,ct2,ct3,dct1,dct2,dct3,ddct1,ddct2,'
            'ddct3,amount1,amount2,amount3\n')
        for item in data:
            vals = [item.day, item.delay, item.region, item.gene]
            vals.extend(pad_to(item.reps, 3))
            vals.extend(pad_to(item.dct, 3))
            vals.extend(pad_to(item.ddct, 3))
            vals.extend(pad_to(item.amount, 3))

            fh.write(','.join(map(str, vals)))
            fh.write('\n')


def format_csv(data):
    output = StringIO()
    output.write('Sample,Target,Cq\n')
    for item in data:
        for val in item.reps:
            output.write('r{},{},{}\n'.format(item.row, item.gene, val))
    output.seek(0)
    return output


def select_housekeepr(csv_buffer, sample_value):
    df = pd.read_csv(csv_buffer, sep=',')
    censored = eleven.censor_background(df)
    ranked = eleven.rank_targets(censored, gene_hk, sample_value)
    return ranked


def normalize_to_housekeeping(data, housekeepers):
    norms = defaultdict(list)
    for item in data:
        if item.gene not in housekeepers:
            continue

        norms[item.row].extend(item.reps)

    for row, items in norms.items():
        norms[row] = np.mean(items)

    for item in data:
        item.dct = item.reps - norms[item.row]


def normalize_dct_to_baseline(data, norm_key='delay', norm_val='pre',
                              norm_hash=('gene', 'region')):
    pool = defaultdict(list)
    for item in data:
        if getattr(item, norm_key) != norm_val:
            continue
        key = tuple([getattr(item, k) for k in norm_hash])
        pool[key].extend(item.dct)

    for key, items in pool.items():
        pool[key] = np.mean(items)

    for item in data:
        key = tuple([getattr(item, k) for k in norm_hash])
        item.ddct = item.dct - pool[key]


def compute_amount(data):
    for item in data:
        item.amount = np.power(1 + efficiency[item.gene], -item.ddct)


def plot_data(data, output, x1='delay', x2=None):
    keys = 'gene', 'region', 'day', 'delay'
    selection = {k: get_values(data, k, globals()[k]) for k in keys}
    x1_labels = selection.pop(x1)
    x2_labels = selection.pop(x2, None)
    variables = selection.keys()  # the variables which are constant for a plot

    # find all the combos of the variables we specify beforehand for the plot
    for spec in product(*[selection[var] for var in variables]):
        if x2 is None:
            sorted_data = defaultdict(list)
        else:
            sorted_data = defaultdict(partial(defaultdict, list))

        # first filter the data to only those selected by spec
        curr_data = [
            d for d in data
            if all([getattr(d, k) == val for k, val in zip(variables, spec)])
        ]

        for item in curr_data:
            if x2 is None:
                sorted_data[getattr(item, x1)].append(item)
            else:
                sorted_data[getattr(item, x2)][getattr(item, x1)].append(item)

        with open(join(output, 'binned_data.csv'), 'ab') as fh:
            if x2 is None:
                for key, vals in sorted_data.items():
                    fh.write(','.join(
                        list(spec) + [key] +
                        map(str, [v for val in vals for v in val.amount])))
                    fh.write('\n')
            else:
                for x2_key, x2_vals in sorted_data.items():
                    for x1_key, vals in x2_vals.items():
                        fh.write(','.join(
                            list(spec) + [x1_key, x2_key] +
                            map(str, [v for val in vals for v in val.amount])))
                        fh.write('\n')

        # now, we have the data sorted by x1/x2 and we just need to do the plot
        fig_spec = []
        for v, n in sorted(list(zip(variables, spec)), key=lambda x: x[0]):
            fig_spec.append('{}; {}'.format(v, n))
        fig_spec = ', '.join(fig_spec)

        fig, ax = plt.subplots(figsize=(18, 10))
        try:
            if x2 is None:
                show_plot1(ax, sorted_data, fig_spec, x1, x1_labels)
            else:
                show_plot2(ax, sorted_data, fig_spec, x1, x1_labels, x2_labels)
        except Exception as e:
            print e
            continue
        plt.tight_layout()
        fig.savefig(join(output, '{}.png'.format(fig_spec)))
        plt.close()


def show_plot1(ax, data, fig_spec, x1label, x1_labels):
    res = np.zeros(len(x1_labels))
    err = np.zeros(len(x1_labels))

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
    ax.set_ylabel('Amount')
    ax.set_xlabel(x1label.capitalize())
    ax.set_xticklabels(x1_labels)
    ax.set_xticks(np.arange(len(x1_labels)))
    ax.legend((rects1[0], ), (x1label, ))
    ax.set_title(fig_spec)


def show_plot2(ax, data, fig_spec, x1label, x1_labels, x2_labels):
    rects = [None, ] * len(x2_labels)
    width = (1 - .2) / float(len(x2_labels))
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / len(x2_labels)) for i in range(len(x2_labels))]

    for i, (label2, color) in enumerate(zip(x2_labels, colors)):
        if label2 not in data:
            raise Exception('Cannot find "{}" for {}'.format(label2, fig_spec))

        res = np.zeros(len(x1_labels))
        err = np.zeros(len(x1_labels))
        x1_data = data[label2]

        for l, label in enumerate(x1_labels):
            if label not in x1_data:
                raise Exception('Cannot find "{}"'.format(label))

            curr_items = x1_data[label]
            items = [v for item in curr_items for v in item.amount]
            assert curr_items
            assert items
            res[l] = np.mean(items)
            err[l] = stats.sem(items)

        rects[i] = ax.bar(
            np.arange(len(x1_labels)) + i * width, res, width, color=color,
            ecolor='k', yerr=err)

    ax.set_ylabel('Amount')
    ax.set_xlabel(x1label.capitalize())
    ax.set_xticklabels(x1_labels)
    ax.set_xticks(np.arange(len(x1_labels)) + 2 * width)
    ax.legend([rect[0] for rect in rects], x2_labels)
    ax.set_title(fig_spec)


if __name__ == '__main__':
    filename = r'C:\Users\Matthew Einhorn\Desktop\20150903a_Analysis_JP.csv'
    output = r'C:\Users\Matthew Einhorn\Desktop\Michelle\figures'
    data = read_data(filename)
    data = filter_data(data, gene=gene_hk + gene, region=region[:-1])

    buff = format_csv(data)
    print(select_housekeepr(buff, 'r{}'.format(data[0].row)))

    normalize_to_housekeeping(data, housekeepers=gene_hk[:1])
    normalize_dct_to_baseline(data, norm_key='delay', norm_val='pre',
                              norm_hash=('gene', 'region'))
    compute_amount(data)

    data = filter_data(data)
    dump_data(data, join(output, 'processed_data.csv'))
    plot_data(data, output, x2='day')
