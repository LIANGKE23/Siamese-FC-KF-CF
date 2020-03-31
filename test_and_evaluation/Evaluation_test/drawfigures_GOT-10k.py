from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import sys

def plot_curves(report_dir,report_files1,report_files2, extension='.png'):

    with open(report_file1) as f:
        performance1 = json.load(f)
    with open(report_file2) as f:
        performance2 = json.load(f)

    succ_file = os.path.join(report_dir, 'success_plot_GOT-10k' + extension)
    key = 'overall'

    # sort trackers by AO
    tracker_names1 = list(performance1.keys())
    aos1 = [t[key]['ao'] for t in performance1.values()]
    inds1 = np.argsort(aos1)[::-1]
    tracker_names1 = [tracker_names1[i] for i in inds1]
    tracker_names2 = list(performance2.keys())
    aos2 = [t[key]['ao'] for t in performance2.values()]
    inds2 = np.argsort(aos2)[::-1]
    tracker_names2 = [tracker_names2[i] for i in inds2]

    # markers
    markers = ['-', '--', '-.']
    markers = [c + m for m in markers for c in [''] * 10]

    # plot success curves
    thr_iou = np.linspace(0, 1, 101)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(tracker_names1):
        line, = ax.plot(thr_iou,
                        performance1[name][key]['succ_curve'],
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (
            name, performance1[name][key]['ao']))

    for i, name in enumerate(tracker_names2):
        line, = ax.plot(thr_iou,
                        performance2[name][key]['succ_curve'],
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (
            name, performance2[name][key]['ao']))

    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='lower left',
                       bbox_to_anchor=(0., 0.))

    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Overlap threshold',
           ylabel='Success rate',
           xlim=(0, 1), ylim=(0, 1),
           title='Success plots on GOT-10k')
    ax.grid(True)
    fig.tight_layout()

    # control ratio
    # ax.set_aspect('equal', 'box')

    print('Saving success plots to', succ_file)
    fig.savefig(succ_file,
                bbox_extra_artists=(legend,),
                bbox_inches='tight',
                dpi=300)

if __name__ == '__main__':
    report_dir = "E:\PSUThirdSemester\CSE586ComputerVision\Term-Project1\Pythonversion\siamfc-pytorch-master/test_and_evaluation\Evaluation_test\REPORT_COMPARE"
    report_file1 = 'E:\\PSUThirdSemester\\CSE586ComputerVision\\Term-Project1\\Pythonversion\\siamfc-pytorch-master\\test_and_evaluation\\Evaluation_test\\reports\\GOT-10k\\SiamFC\\performance.json'
    report_file2 = 'E:\\PSUThirdSemester\\CSE586ComputerVision\\Term-Project1\\Pythonversion\\siamfc-pytorch-master\\test_and_evaluation\\Evaluation_test\\reports_KF_CF\\GOT-10k\\SiamFC_KF_CF\\performance.json'
    plot_curves(report_dir, report_file1, report_file2)