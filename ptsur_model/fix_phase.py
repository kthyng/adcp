'''Trying to fix Point Sur heading issues.

The heading correction (hcorr) is not good, so I look at phase vs. heading
for the three existing Point Sur cruises, reduce the data in an ad hoc manner,
fit a sine wave to it since it looks that way and Jules said it made sense,
so that I have the phase as a function of heading.

After that, I output the decimal days and phase. This can be used in the future
with data gathered on the Point Sur using their navigation feed.

This is necessary because the heading data is bad and the heading correction is
wrong. We can instead get the phase correct from the bottom tracking.

Kristen Thyng Oct 2017
'''

from pycurrents.adcp.quick_mpl import Btplot
import pandas as pd
import os
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

modeldfname = 'model_df.csv'
modelfitname = 'model_fit.txt'


def sine_fit(x, freq, amplitude, phase, offset):
    '''sine wave function.'''

    return np.sin(x * np.deg2rad(freq) + np.deg2rad(phase)) * amplitude + offset


def load_data(postprocloc):
    '''Load amp, phase, heading data from cruise into a dataframe.

    Input directory postprocloc is where postprocessing is occurring, e.g.
    `ps18_09_l1_postproc/wh300.unrotated`.
    '''

    BT = Btplot()
    BT(btm_filename=postprocloc + '/cal/botmtrk/a_ps.btm',
       ref_filename=postprocloc + '/cal/botmtrk/a_ps.ref',
       cruiseid = 'ADCP',
       min_depth = 10,
       max_depth = 300,
       printformats = 'png',
       proc_yearbase='2017')
    plt.close(plt.gcf())

    # create dataframe for amp/ph data
    uv = pd.DataFrame(index=BT.dayuv, data={'phase': BT.ph, 'amp': BT.a})
    # bottom track dataframe with own indices
    bt = pd.DataFrame(index=BT.day_bt, data={'heading': BT.heading_bt})
    scn = pd.read_table(postprocloc + '/cal/rotate/scn.hdg', index_col=0,
                          delim_whitespace=True, header=None, comment='%',
                          names=['mean heading', 'last heading', 'correction'])
    # hcorr = pd.read_table(leg + '/proc/wh300/cal/rotate/ens_hcorr.ang', index_col=0,
    #                       delim_whitespace=True, header=None, names=['hcorr'])
    # all indices included
    df = scn.join(uv, how='outer').join(bt, how='outer').interpolate()
    df = df.reindex(scn.index)  # just scn indices
    dfnonan = df[~df.isnull().sum(axis=1).astype(bool)]  # with no nans

    return dfnonan


def find_model():
    '''Original finding of model between phase and heading.

    This does not need to be repeated unless the comparison between the present
    data and this model doesn't match well. Check this in compare_with_model().'''

    # use data from all three available legs to create this model
    legs = ['ps18_09_l1']#, 'ps18_09_l2', 'ps18_09_l3']
    devdir = 'wh300.unrotated'

    # loop through legs and get data into dataframes
    dfs = []; dfnns = []
    for leg in legs:

        df = load_data(leg + '_postproc/' + devdir)

        dfnns.append(df)
        # dfs.append(df)

    # combine these three good dfs in one. df has nan's and dfnn does not.
    # df = pd.concat(dfs)
    dfnn = pd.concat(dfnns)

    # look at phase vs. heading for the three cruises to see if there is a relationship
    colors = ['k', 'darkcyan', 'purple']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, df in enumerate(dfs):
        ax.plot(dfnn['heading'], dfnn['phase'], 'o', color=colors[i], alpha=0.5, label=legs[i])
    ax.set_xlabel('heading')
    ax.set_ylabel('phase')

    # Ad hoc: remove visually bad info: when phase < 1, and phase > 3
    dfnn[(dfnn['phase']<1) | (dfnn['phase']>3)] = np.nan
    dfnn = dfnn[~dfnn.isnull().sum(axis=1).astype(bool)]
    # also remove weird blobs above and below visible sine wave (and remove nans)
    mask = (dfnn['phase']<2.5) & (dfnn['heading']>10) & (dfnn['heading']<90)
    dfnn[mask] = np.nan
    dfnn = dfnn[~dfnn.isnull().sum(axis=1).astype(bool)]
    mask = (dfnn['phase']>2) & (dfnn['heading']<-50)
    dfnn[mask] = np.nan
    dfnn = dfnn[~dfnn.isnull().sum(axis=1).astype(bool)]

    # look at combined data with bad data removed
    ax.plot(dfnn['heading'], dfnn['phase'], 'o', label='data', color='r', alpha=0.5)

    # do curve fit of remaining data as a sine wave
    fit = curve_fit(sine_fit, dfnn['heading'].values, dfnn['phase'].values)
    data_fit = sine_fit(dfnn['heading'].values, *fit[0])
    ax.plot(dfnn['heading'].values, data_fit, '.', label='fitting', color='m', lw=3)
    ax.legend(loc='best', numpoints=1)
    ax.set_ylim(1,3)
    fig.savefig('model.png', bbox_inches='tight')

    dfnn.to_csv(modeldfname)
    np.savetxt(modelfitname, fit[0])



def use_model(postprocloc):
    '''Compare models.

    Compare the relationship between phase and bottom tracking heading as
    calculated from the present data set and from ps18_09_l1.
    Input directory postprocloc is where postprocessing is occurring, e.g.
    `ps18_09_l1_postproc/wh300.unrotated`.
    '''

    # read in model
    fit = np.loadtxt(modelfitname)

    # read in data
    df = load_data(postprocloc)

    # read in information for current data
    scn = pd.read_table(postprocloc + '/cal/rotate/scn.hdg', index_col=0,
                          delim_whitespace=True, header=None, comment='%',
                          names=['mean heading', 'last heading', 'correction'])

    # apply model to calculate new phase correction column
    scn['phase'] = sine_fit(scn['last heading'].values, *fit)

    # Plot up to compare check with data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df['heading'], df['phase'], 'o', alpha=0.5, label='data')
    ax.plot(scn['last heading'], scn['phase'], 'o', alpha=0.5, label='model')
    ax.set_xlabel('heading')
    ax.set_ylabel('phase')
    ax.set_ylim(0.5,3.5)

    # Plot back onto original calibration plot to compare
    BT = Btplot()
    BT(btm_filename=postprocloc + '/cal/botmtrk/a_ps.btm',
       ref_filename=postprocloc + '/cal/botmtrk/a_ps.ref',
       cruiseid = 'ADCP',
       min_depth = 10,
       max_depth = 300,
       printformats = 'png',
       proc_yearbase='2017', load_only=True)

    fig2 = plt.figure(figsize=(12,4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(BT.dayuv, BT.ph, 'b.')
    ax2.plot(df.index, df['phase'].values, 'gx')
    # ax2.axis('tight')
    ax2.set_ylim(0.5,3.5)
    ax2.set_xlabel('Decimal Day')
    ax2.set_ylabel('Degrees')
    # just show part of a day
    ax2.set_xlim(BT.dayuv.min(), BT.dayuv.min()+0.3)

    # save phase correction into files,
    # one for positive and negative application of correction
    # positive angle
    fname = postprocloc + '/cal/rotate/rotate_btphase_positive.ang'
    scn['phase'].to_csv(fname, sep=' ')
    # negative angle
    fname = postprocloc + '/cal/rotate/rotate_btphase_negative.ang'
    (-scn['phase']).to_csv(fname, sep=' ')
