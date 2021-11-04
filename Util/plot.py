import matplotlib.pyplot as plt
import numpy as np
from glob import *


# plot the data valuation results

def plot_util(path, round_group_sv, cumu_group_sv):
    if NOISE_ADD == True:
        fig_title_round = path + 'Round-ND.png'
        fig_title_cumu = path + 'Cumu-ND.png'
    elif UNBALANCE == True:
        fig_title_round = path + 'Round-UD.png'
        fig_title_cumu = path + 'Cumu-UD.png'
    elif NON_IID == True:
        fig_title_round = path + '/Round-noniid.png'
        fig_title_cumu = path + 'Cumu-noniid.png'
    elif NOISE_ADD_LATER == True:
        fig_title_round = path + 'Round-ndlater.png'
        fig_title_cumu = path + 'Cumu-ndlater.png'
    elif UNBALANCED_LATER == True:
        fig_title_round = path + 'Round-udlater.png'
        fig_title_cumu = path + 'Cumu-udlater.png'
    else:
        fig_title_round = path + 'Round-OD.png'
        fig_title_cumu = path + 'Cumu-OD.png'


    font1 = {
        #'family' : 'Times New Roman',
        'weight': 'normal',
        'size': 18,
    }

    font2 = {
        #'family' : 'Times New Roman',
        'weight': 'normal',
        'size': 16,
    }

    x_axix = [i for i in range(NUM_ROUND)]

    plt.figure()
    plt.plot(x_axix, round_group_sv[0], '-o', color='#96ceb4', label='P1')
    plt.plot(x_axix, round_group_sv[1], '-s', color='#a696c8', label='P2')
    plt.plot(x_axix, round_group_sv[2], '-p', color='#d9534f', label='P3')
    plt.plot(x_axix, round_group_sv[3], '-v', color='#ffad60', label='P4')
    plt.plot(x_axix, round_group_sv[4], '-*', color='#05445c', label='P5')

    plt.xticks(np.arange(1, NUM_ROUND, 2))
    plt.legend(ncol=5, loc=1)
    plt.xlabel('Round #', font1)
    plt.ylabel('Round SV', font1)
    plt.grid(axis='y', linestyle='-.')
    plt.ylim(-1, 1.5)
    plt.tick_params(labelsize=12)
    plt.savefig(fig_title_round, dpi=300, bbox_inches='tight')
    #plt.show()

    plt.figure()
    plt.plot(x_axix, cumu_group_sv[0], '-o', color='#96ceb4', label='P1')
    plt.plot(x_axix, cumu_group_sv[1], '-s', color='#a696c8', label='P2')
    plt.plot(x_axix, cumu_group_sv[2], '-p', color='#d9534f', label='P3')
    plt.plot(x_axix, cumu_group_sv[3], '-v', color='#ffad60', label='P4')
    plt.plot(x_axix, cumu_group_sv[4], '-*', color='#05445c', label='P5')

    plt.xticks(np.arange(1, NUM_ROUND, 2))
    plt.legend(ncol=5, loc=1)
    plt.xlabel('Round #', font1)
    plt.ylabel('Cumulative SV', font1)
    if NOISE_ADD == False:
        plt.ylim(-4, 4)
    plt.tick_params(labelsize=12)
    plt.grid(axis='y', linestyle='-.')
    plt.savefig(fig_title_cumu, dpi=300, bbox_inches ='tight')
    #plt.show()
