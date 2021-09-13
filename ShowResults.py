# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 2:00 下午
# @Author  : Yushuo Wang
# @FileName: ShowResults.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import torch
import numpy as np
import matplotlib.pyplot as plt

name = '124_complete'
pthfile2 = r'/mnt/6faed242-81e6-4dae-a66f-77b750a0b1c9/workspace/yushuo/workspace/pytorch-vqvae/results/2_vqvae_data_sun_sep_12_14_10_52_2021.pth'
pthfile1 = r'/mnt/6faed242-81e6-4dae-a66f-77b750a0b1c9/workspace/yushuo/workspace/pytorch-vqvae/results/1_vqvae_data_sun_sep_12_14_34_13_2021.pth'
pthfile4 = '/mnt/6faed242-81e6-4dae-a66f-77b750a0b1c9/workspace/yushuo/workspace/pytorch-vqvae/results/4_vqvae_data_sun_sep_12_22_15_14_2021.pth'
def showResults(pthfile):
    results = torch.load(pthfile)
    log_interval = 50
    round = (results['results']['n_updates'])//log_interval + 1
    RECON = []
    Loss = []
    Perplexity = []

    for i in range((results['results']['n_updates'])//log_interval + 1):
        RECON.append(np.mean(results['results']["recon_errors"][i*log_interval:(i+1)*log_interval]))
        Loss.append(np.mean(results['results']["loss_vals"][i*log_interval:(i+1)*log_interval]))
        Perplexity.append(np.mean(results['results']["perplexities"][i*log_interval:(i+1)*log_interval]))

        print('Update #', i, 'Recon Error:',
              np.mean(results['results']["recon_errors"][i*log_interval:(i+1)*log_interval]),
              'Loss', np.mean(results['results']["loss_vals"][i*log_interval:(i+1)*log_interval]),
              'Perplexity:', np.mean(results['results']["perplexities"][i*log_interval:(i+1)*log_interval]))
    return RECON, Loss, Perplexity, round

# f = open(pthfile4)
# lines = f.readlines()
# RECON4 = []
# Loss4 = []
# Perplexity4 = []
# for line in lines:
#     res = list(line.strip().split(' '))
#     RECON4.append(float(res[5]))
#     Loss4.append(float(res[7]))
#     Perplexity4.append(float(res[-1]))

RECON1, Loss1, Perplexity1, round = showResults(pthfile1)
RECON2, Loss2, Perplexity2, _ = showResults(pthfile2)
RECON4, Loss4, Perplexity4, _ = showResults(pthfile4)

def loss_ln(loss):
    res_ = np.array(loss)
    res = np.log(res_ + 1)
    return res

Loss1 = loss_ln(Loss1)
Loss2 = loss_ln(Loss2)
Loss4 = loss_ln(Loss4)

plt.plot(np.arange(round), RECON1)
plt.plot(np.arange(round), RECON2)
plt.plot(np.arange(round), RECON4)
plt.title('recon_errors')
plt.legend(['1', '2', '4'], loc='upper right')
plt.savefig('results/imgs/' + name + '_recon_errors.jpg')
plt.show()

plt.plot(np.arange(round), Loss1)
plt.plot(np.arange(round), Loss2)
plt.plot(np.arange(round), Loss4)
# plt.ylim(0, 150)
plt.title('loss_vals')
plt.legend(['1', '2', '4'], loc='upper right')
plt.savefig('results/imgs/' + name + '_ln_loss_vals.jpg')
plt.show()

plt.plot(np.arange(round), Perplexity1)
plt.plot(np.arange(round), Perplexity2)
plt.plot(np.arange(round), Perplexity4)
plt.title('perplexities')
plt.legend(['1', '2', '4'], loc='upper left')
plt.savefig('results/imgs/' + name + '_perplexities.jpg')
plt.show()

_ = 1