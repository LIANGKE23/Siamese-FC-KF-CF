from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import sys

def plot_curves(report_dir,report_file1,report_file2):

    # load pre-computed performance
    with open(report_file1) as f:
        performance1 = json.load(f)
    with open(report_file2) as f:
        performance2 = json.load(f)

    succ_file = os.path.join(report_dir, 'success_plots_UAV123.png')
    prec_file = os.path.join(report_dir, 'precision_plots_UAV123.png')
    key = 'overall'

    # markers
    markers = ['-', '--', '-.']
    markers = [c + m for m in markers for c in [''] * 10]

    # sort trackers by success score
    tracker_names1 = list(performance1.keys())
    tracker_names2 = list(performance2.keys())
    succ1 = [t[key]['success_score'] for t in performance1.values()]
    inds1 = np.argsort(succ1)[::-1]
    tracker_names1 = [tracker_names1[i] for i in inds1]
    succ2 = [t[key]['success_score'] for t in performance2.values()]
    inds2 = np.argsort(succ2)[::-1]
    tracker_names2 = [tracker_names2[i] for i in inds2]

    # plot success curves
    thr_iou = np.linspace(0, 1, 21)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(tracker_names1):
        line, = ax.plot(thr_iou,
                        performance1[name][key]['success_curve'],
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, performance1[name][key]['success_score']))

    for i, name in enumerate(tracker_names2):
        line, = ax.plot(thr_iou,
                        performance2[name][key]['success_curve'],
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, performance2[name][key]['success_score']))

    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='center left',
                       bbox_to_anchor=(1, 0.5))

    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Overlap threshold',
           ylabel='Success rate',
           xlim=(0, 1), ylim=(0, 1),
           title='Success plots of OPE')
    ax.grid(True)
    fig.tight_layout()

    print('Saving success plots to', succ_file)
    fig.savefig(succ_file,
                bbox_extra_artists=(legend,),
                bbox_inches='tight',
                dpi=300)

    # sort trackers by precision score
    tracker_names1 = list(performance1.keys())
    prec1 = [t[key]['precision_score'] for t in performance1.values()]
    inds1 = np.argsort(prec1)[::-1]
    tracker_names1 = [tracker_names1[i] for i in inds1]

    tracker_names2 = list(performance2.keys())
    prec2 = [t[key]['precision_score'] for t in performance2.values()]
    inds2 = np.argsort(prec2)[::-1]
    tracker_names2 = [tracker_names2[i] for i in inds2]

    # plot precision curves
    thr_ce = np.arange(0, 51)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(tracker_names1):
        line, = ax.plot(thr_ce,
                        performance1[name][key]['precision_curve'],
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, performance1[name][key]['precision_score']))

    for i, name in enumerate(tracker_names2):
        line, = ax.plot(thr_ce,
                        performance2[name][key]['precision_curve'],
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, performance2[name][key]['precision_score']))

    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='center left',
                       bbox_to_anchor=(1, 0.5))

    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Location error threshold',
           ylabel='Precision',
           xlim=(0, thr_ce.max()), ylim=(0, 1),
           title='Precision plots of OPE')
    ax.grid(True)
    fig.tight_layout()

    print('Saving precision plots to', prec_file)
    fig.savefig(prec_file,bbox_extra_artists=(legend,), dpi=300)



if __name__ == '__main__':
    report_dir = "E:\PSUThirdSemester\CSE586ComputerVision\Term-Project1\Pythonversion\siamfc-pytorch-master/test_and_evaluation\Evaluation_test\REPORT_COMPARE"
    report_file1 = 'E:\\PSUThirdSemester\\CSE586ComputerVision\\Term-Project1\\Pythonversion\\siamfc-pytorch-master\\test_and_evaluation\\Evaluation_test\\reports\\UAV123\\SiamFC\\performance.json'
    report_file2 = 'E:\\PSUThirdSemester\\CSE586ComputerVision\\Term-Project1\\Pythonversion\\siamfc-pytorch-master\\test_and_evaluation\\Evaluation_test\\reports_KF_CF\\UAV123\\SiamFC_KF_CF\\performance.json'
    plot_curves(report_dir, report_file1, report_file2)