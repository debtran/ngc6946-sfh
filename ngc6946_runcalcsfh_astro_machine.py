"""
Functions for executing multiple runs of the calcsfh command line program.

Included is a wrapper for the calcsfh command line program, and some Av,
dAv search algorithms to control where calcsfh is run in Av, dAv space.
These functions are written very generally and are not intended to be
customized. All customization (logging, file organization, calcsfh
parameters, etc.) should should be d one elsewhere, e.g., main().

The required command line arguments depend on the mode of operation::

  {name} {proj_dir} model pattern {Av0} {dAv0} {d_Av} {d_dAv} {d_Av0} {d_dAv0}
  {name} {proj_dir} model grid {Av1} {dAv1} {Av2} {dAv2} {nAv} {ndAv}
  {name} {proj_dir} model list {pointfile}

`name` could be an object or region name being analyzed, and pattern, grid,
and list refer to functions that manage the calcsfh runs. The arguments are
described in their respective functions.

example on how to run: 
python /astro/users/detran/MATCH/test.py 2265 /astro/users/detran/MATCH/NGC6946 Padua2006_CO_AGB pattern 0.8 0.0 0.2 0.2 0.05 0.05

"""

import numpy as np
import os
import shutil
import subprocess
import sys
import time
from astropy.io import ascii
import pandas as pd
import errno
from lxml import etree

def format_table(table_xml, level=1, indent='  '):
    """
    XML pretty-print formatter for TABLE elements.

    """
    table_xml.text = '\n' + level*indent
    coldef_xml, data_xml = table_xml

    coldef_xml.text = '\n' + (level+1)*indent
    for c_xml in coldef_xml:
        c_xml.tail = '\n' + (level+1)*indent
    c_xml.tail = '\n' + (level)*indent
    coldef_xml.tail = '\n' + (level)*indent

    data_xml.text = '\n' + (level+1)*indent
    if len(data_xml):
        for r_xml in data_xml:
            r_xml.tail = '\n' + (level+1)*indent
        r_xml.tail = '\n' + (level)*indent
    data_xml.tail = '\n' + (level-1)*indent
    table_xml.tail = '\n'


def safe_mkdir(path):
    """
    A safer version of os.makedirs. Path is only created if it does not exist.

    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def calcsfh(photfile, fakefile, IMF, dmod, Av, res, logZ,
            BF, badfrac, CMD, tbins,
            paramfile=None, sfhfile=None, cmdfile=None, terminalfile=None,
            zinc=None, dAv=None, dAvy=None, models=None):
    """
    Python wrapper for calcsfh in MATCH.

    Most of the function arguments are used to create a calcsfh parameter
    file on the fly before running calcsfh. The remaining arguments are
    used to set up the calcsfh command. Refer to the MATCH README file for
    further information about the parameter file and useful settings for
    the arguments.

    When the calcsfh has finished running, the fit value of the best-fit
    solution is returned along with all content in the .sfh and .sfh.cmd
    output files. All output that calcsfh sends to stdout is piped to
    terminalfile.

    This wrapper has only been tested with MATCH 2.7. The following
    features are not currently supported:

    - Multi-CMD fitting, i.e., obtaining SFHs from photometry for more than
      two filters.
    - Specification of a background CMD.
    - Many other calcsfh flags (e.g., -logterrsig, -mbolerrsig, etc.).

    Parameters
    ----------
    photfile : string
        Path to the input photometry file.
    fakefile : string
        Path to the fake star photometry file.
    IMF : {'Kroupa', 'Salpeter', float}
        IMF slope. 'Kroupa' is equivalent to nothing, 'Salpeter' is equivalent
        to 1.35. (not -1.35).
    dmod : list or tuple
        Minimum and maximum distance modulus values, (min, max).
    Av : list or tuple
        Minimum and maximum foreground V-band extinction values,
        (min, max).
    res : float
        Resolution for dmod and Av (the same value is used for both).
    logZ : list or tuple
        Minimum, maximum, and resolution values for metallicity (solar=0),
        (min, max, res).
    BF : int or float
        Binary fraction.
    badfrac : list or tuple
        Fraction of bad detections at the top and bottom of the CMD,
        (top, bottom).

    CMD : dict
        The CMD dict contains the following keys:

        V : dict
            A filter dictionary for the bluer ("V") filter. Contains the
            keys 'name' for the filter name, and 'min' and 'max' for the
            bright and faint magnitude limits of the CMD. E.g., ::

              filter1 = {'name': 'WFC475W', 'min': 16.0, 'max': 27.0}.

        I : dict
            Same as V, but for the redder ("I") filter.
        Vres : float
            Resolution (bin size) of the V filter magnitude in the CMD.
        V-I : list or tuple
            Minimum, maximum, and resolution (bin size) values for V-I
            color in the CMD, (min, max, res).
        fake_sm : int or float
            Factor used to improve fake star statistics.
        exclude_gates : list, optional
            If specified (or if not None), all exclude gates in the list
            will be applied to the CMD. A single gate is a list of four
            points::

              exgate = [(color1, mag1), (color2, mag2),
                        (color3, mag3), (color4, mag4)]

            Clockwise vs. counterclockwise order does not matter.
        combine_gates : list, optional
            Same as exclude_gates, but for combine gates.

    tbins : list
        List of time bins, one bin per element. Each element must be a list
        containing at least the young and old age limits of the bin.
        Additional values may be specified after the old age limit, e.g., a
        third number to fix the SFR in a particular bin.
    paramfile : string, optional
        Path to the calcsfh parameter file created based on the function
        arguments. Default is photfile + '.par'.
    sfhfile : string, optional
        Path to the .sfh output file. Default is photfile + '.sfh'.
    cmdfile : string, optional
        Path to the .cmd output file. Default is photfile + '.sfh.cmd'
    terminalfile : string, optional
        Path to the calcsfh terminal output file. Default is
        photfile + '.out'.
    zinc : list or tuple, optional
        Constrain metallicity to increase with time with a list of minimum and
        maximum initial and final metallicity vaues, (initial_min, initial_max,
        final_min, final_max). Default is None (no constraint).
    dAv : float, optional
        Maximum differential extinction. Default is None (no differential
        extinction).
    dAvy : float, optional
        Maximum additional differential extinction for young stars. Default
        is None (no additional differential extinction).
    models : {'PADUA_AGB', etc.}, optional
        Model selection flag. If None, the calcsfh internal default Padua
        2006 models are used.
        
    PARAM file format
    -------
     Setup calcsfh parameters
     The format of the parameter file is as follows:
     IMF dmod_min dmod_max res Avmin Avmax dAv
     logZmin logZmax dlogZ Zinitial_min Zinitial_max Zpresent_min Zpresent_max (using -zinc flag)
     BF Bad0 Bad1
     Ncmds
     Vstep V-Istep fake_sm V-Imin V-Imax V,I  (per CMD)
     Vmin Vmax V                              (per filter)
     Imin Imax I                              (per filter)
     Nexclude_gates exclude_gates Ncombine_gates combine_gates (per CMD)
     Ntbins
     To Tf (for each time bin)

    Returns
    -------
    fit : float
        Fit value of best-fit solution from calcsfh.
    sfh : string
        All content from the output .sfh file (large amount of data).
    cmd : string
        All content from the output .cmd file (large amount of data).

    """
    if IMF == 'Kroupa':
        #IMF = '-1.0'
        IMF = ' ' #Edit on July 27, 2020. Newest version of MATCH wants NOTHING, not a -1 to indicate Kroupa IMF
        imf_flag = '-kroupa'
    elif IMF == 'Salpeter':
        IMF = '1.35'
        imf_flag = None
    else:
        IMF = '{0:.2f}'.format(IMF)
        imf_flag = None
    if paramfile is None:
        paramfile = '{0:s}.par'.format(photfile)
    if sfhfile is None:
        sfhfile = '{0:s}.sfh'.format(photfile)
    if cmdfile is None:
        cmdfile = '{0:s}.sfh.cmd'.format(photfile)
    if terminalfile is None:
        terminalfile = '{0:s}.out'.format(photfile)

    # Write the parameter file
    with open(paramfile, 'w') as f:
        # IMF, dmod, Av
        f.write('{0:s} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2f} {3:.2f}\n'
                .format(IMF, dmod[0], dmod[1], res, Av[0], Av[1]))
        # logZ, zinc
        line = '{0:.1f} {1:.1f} {2:.2f}'.format(*logZ)
        if zinc is not None:
            line = '{0:s} {1:.1f} {2:.1f} {3:.1f} {4:.1f}\n'.format(line, *zinc)
        else:
            line = '{0:s}\n'.format(line)
        f.write(line)

        # binary fraction, bad fractions - still unclear what bad fraction is, just very small value?
        f.write('{0:.2f} {1:.6f} {2:.6f}\n'.format(BF, badfrac[0], badfrac[1]))

        # CMD info
        # number of cmds
        N = 1 
        f.write('{0:d}\n'.format(N))
        # V step size, color stepsize, fake stars smoothing param, color min, colormax, V name, I name 
        f.write('{0:.2f} {1:.2f} {2:.0f} {3:.2f} {4:.2f} {5:s},{6:s}\n'
                .format(CMD['Vres'], CMD['V-I'][2], CMD['fake_sm'],
                        CMD['V-I'][0], CMD['V-I'][1],
                        CMD['V']['name'], CMD['I']['name']))

        # Filter info
        # Vmin, Vmax, Vname
        # Imin, Imax, Iname
        f.write('{0:.2f} {1:.2f} {2:s}\n'
                .format(CMD['V']['min'], CMD['V']['max'], CMD['V']['name']))
        f.write('{0:.2f} {1:.2f} {2:s}\n'
                .format(CMD['I']['min'], CMD['I']['max'], CMD['I']['name']))

        # Gate info
        #ML Edit July 27, 2020 - python 3 doesn't use has_key
        #if CMD.has_key('exclude_gates') and CMD['exclude_gates'] is not None:
        
        if ('exclude_gates' in CMD)and (CMD['exclude_gates'] is not None):
            N = len(CMD['exclude_gates'])
            line = '{0:d}'.format(N)
            for gate in CMD['exclude_gates']:
                for x, y in gate:
                    line = ('{0:s} {1:.2f} {2:.2f}'.format(line, x, y))
        else:
            line = '0'
         
        #Edit July 27, 2020 - python 3 doesn't use has_key
        #if CMD.has_key('combine_gates') and CMD['combine_gates'] is not None:
        if ('combine_gates' in CMD)and (CMD['combine_gates'] is not None):
            N = len(CMD['combine_gates'])
            line = '{0:s} {1:d}'.format(line, N)
            for gate in CMD['combine_gates']:
                for x, y in gate:
                    line = ('{0:s} {1:.2f} {2:.2f}'.format(line, x, y))
        else:
            line = '{0:s} 0'.format(line)
        f.write('{0:s}\n'.format(line))

        # Time bins
        N = len(tbins)
        f.write('{0:d}\n'.format(N))
        for tbin in tbins:
            if len(tbin) == 3:
                line = '{0:.2f} {1:.2f} {2:.4e}\n'.format(*tbin)
            else:
                line = '{0:.2f} {1:.2f}\n'.format(*tbin)
            f.write(line)


    # Construct the calcsfh command
    calcsfh = '/astro/apps7/opt/match2.7/bin/calcsfh' #ML Edit July 27, 2020. Updated path to astro MATCH installation
    
    command = ('{0:s} {1:s} {2:s} {3:s} {4:s}'
               .format(calcsfh, paramfile, photfile, fakefile, sfhfile))
    if zinc is not None:
        command = '{0:s} -zinc'.format(command)
    #ML Update July 28, 2020 - match2.7 default is PADUA2006_CO_AGB, so no flag needs to be added
    if models is not None:
        command = '{0:s} -{1:s}'.format(command, models)
    if dAv is not None:
        command = '{0:s} -dAv={1:.2f}'.format(command, dAv)
    if dAvy is not None:
        command = '{0:s} -dAvy={1:.2f}'.format(command, dAvy)
    #ML Edit July 27, 2020: If IMF = Kroupa, must add -kroupa flag to calcsfh command
    if imf_flag == '-kroupa':
        command = '{0:s} -kroupa'.format(command)
    command = '{0:s} > {1:s}'.format(command, terminalfile)

    # Run calcsfh
    print('calcsfh command=',command)
    subprocess.call(command, shell=True)
    time.sleep(5)

    # Get fit value from calcsfh terminal output file
    with open(terminalfile, 'r') as f:
        for line in f:
            if line.startswith('Best fit: '):
                fit = float(line.split()[4].split('=')[1])

    # Get contents from .sfh and .sfh.cmd files
    with open(sfhfile, 'r') as f:
        sfh = f.read()
    with open(cmdfile, 'r') as f:
        sfhcmd = f.read()

    return fit, sfh, sfhcmd

#if you want grid or pointlist search, refer back to Margaret's pipeline.

def pattern_search(func, xy0, dxy, dxy0, divisor=2, xlim=None, ylim=None):
    """
    Find x and y that minimizes func using a 2-d pattern search.

    The pattern is a 9-point (3x3) array, starting with x0,y0 in the center
    and the eight surrounding points on a 2*dx by 2*dy square. func is
    evaluated at each point in the pattern and the point with the smallest
    result becomes the new central point. If the central point has the
    smallest result, then the step size is reduced; if the step size is
    already at the smallest allowed value, then the central point is the
    minimum of func.

    Parameters
    ----------
    func : function
        The function to be minimized.
    xy0 : tuple
        Initial seed point for the search as a pair of coordinates, (x, y).
    dxy : float or tuple
        Initial step size for point pattern. A single value is applied to
        both x and y; a tuple can be used for different x and y step sizes,
        (dx, dy).
    dxy0 : float or tuple
        Minimum value(s) for dxy.
    divisor : int or float, optional
        Amount by which to divide the step size (default is 2, i.e., divide
        step size in half).
    xlim, ylim : list or tuple, optional
        Two-element list containing the minimum and maximum values of x and
        y to search within (default is None). To set only one extreme of a
        coordinate and leave the other unconstrained, set the unconstrained
        extreme to +/-np.inf.

    Returns
    -------
    x, y, z :
        x and y coordinates where func is minimized, and the value of func
        at that point.

    """
    try:
        x0, y0 = xy0
    except TypeError:
        x0, y0 = xy0, xy0

    try:
        dx, dy = dxy
    except TypeError:
        dx, dy = dxy, dxy

    try:
        dx0, dy0 = dxy0
    except TypeError:
        dx0, dy0 = dxy0, dxy0

    if xlim is None:
        xlim = (-np.inf, np.inf)
    if ylim is None:
        ylim = (-np.inf, np.inf)

    while 1:
        # 9-point pattern
        x_list = np.array([x0, x0, x0+dx, x0+dx, x0+dx,
                           x0, x0-dx, x0-dx, x0-dx])
        y_list = np.array([y0, y0+dy, y0+dy, y0, y0-dy,
                           y0-dy, y0-dy, y0, y0+dy])

        # Keep x and y within limits
        i = ((xlim[0] <= x_list) & (x_list <= xlim[1]) &
             (ylim[0] <= y_list) & (y_list <= ylim[1]))
        x_list, y_list = x_list[i], y_list[i]

        # Evaluate func at each point in the pattern
        z_list = np.array([func(x, y) for x, y in zip(x_list, y_list)])

        # Find minimum point and either move or shrink the point pattern.
        # Break if central point is the minimum in the smallest pattern.
        i = z_list.argmin()
        if (x_list[i], y_list[i]) == (x0, y0):
            dx, dy = dx/divisor, dy/divisor
            if dx < dx0 and dy > dy0:
                dx = dx0
            elif dy < dy0 and dx > dx0:
                dy = dy0
            elif dx < dx0 and dy < dy0:
                break
        else:
            x0, y0 = x_list[i], y_list[i]

    return x_list[i], y_list[i], z_list[i]

def main():
    """
    The behavior of calcsfh and how the results are logged and organized
    are controlled here. The main tasks are,

    - Specify paths to directories and files and create any directories
      that do not yet exist.
    - Define a function, which gets passed to a search algorithm or run
      manager, that wraps around calcsfh to perform logging, retrieve
      results from previous calculations, and organize calcsfh output.
    - Specify the calcsfh parameters and command line options.
    - Call the search algorithm or run manager.

    """
    def func(Av, dAv):
        """
        Wrapper for calcsfh to record and recall fit values (preventing
        redundant calculations) and write .sfh and .cmd files to
        archive_dir.

        """
        # Retrieve prior result or start a new run
        Avstr, dAvstr = '{0:.2f}'.format(Av), '{0:.2f}'.format(dAv)
        if (Avstr, dAvstr) in record:
            fit = float(record[(Avstr, dAvstr)])
        else:
            calcsfhargs = [photfile, fakefile, IMF, dmod, (Av, Av), res, logZ,
                           BF, badfrac, CMD, tbins]
            
        # Files
            paramfile = os.path.join(archive_dir, '{0:s}_{1:s}.par'.format(Avstr,dAvstr))
            sfhfile = os.path.join(archive_dir, '{0:s}_{1:s}.sfh'.format(Avstr,dAvstr))
            cmdfile = os.path.join(archive_dir,'{0:s}.cmd'.format(sfhfile))
            terminalfile = os.path.join(archive_dir, '{0:s}_{1:s}.log'.format(Avstr,dAvstr))
            for filename in [paramfile, sfhfile, cmdfile, terminalfile]:
                safe_mkdir(os.path.dirname(filename))

            calcsfhkwargs = {'paramfile': paramfile, 'sfhfile': sfhfile,
                             'cmdfile': cmdfile, 'terminalfile': terminalfile,
                             'zinc': zinc, 'dAv': dAv, 'dAvy': dAvy,
                             'models': models}

            fit, sfh, sfhcmd = calcsfh(*calcsfhargs, **calcsfhkwargs)

            # Log the result, write sfh and cmd files for archive
            fitstr = '{0:.6f}'.format(fit)
            record[(Avstr, dAvstr)] = fitstr
            r_xml = etree.SubElement(data_xml, 'R')
            for val in [Avstr, dAvstr, fitstr]:
                c_xml = etree.SubElement(r_xml, 'C')
                c_xml.text = val
            format_table(table_xml)
            tree.write(logfile, encoding='UTF-8', xml_declaration=True)

            sfhout = os.path.join(archive_dir,
                                  '{0:s}_{1:s}.sfh'.format(Avstr, dAvstr))
            with open(sfhout, 'w') as f:
                f.write(sfh)

            cmdout = '{0:s}.cmd'.format(sfhout)
            with open(cmdout, 'w') as f:
                f.write(sfhcmd)

        return fit
    
    #everything except for fake stars in one directory for ease. 
    #name = 'NGC6946_2265'
    name = 'NGC6946_'+str(sys.argv[1])
    #August 18, 2020: ML Added proj_dir as an input argument for scripting over multiple SFH regions
    proj_dir = sys.argv[2]
    #print('project directory=',proj_dir)
    #print('cwd =',os.getcwd())
    # Path constants\
    #proj_dir = '/astro/users/detran/MATCH/NGC6946/'
    fake_dir = os.path.join(proj_dir, 'fake')
    phot_dir = os.path.join(proj_dir, 'grids', name)
    archive_dir = os.path.join(phot_dir, 'archive')

    #Read in F275W and F336W completeness (determined by density bin previously) to define the magnitude limits to include on the CMD:
    #Edit 20220610, D. Tran 
    density_comp = pd.read_csv(proj_dir + '/grid_density_completeness.csv')
    f275w_50_comp = density_comp.F275_comp.values[int(sys.argv[1])] #completeness val of the specific grid
    f336w_50_comp = density_comp.F336_comp.values[int(sys.argv[1])]
    
    #point to correct fake stars based on density bin
    density = density_comp.density[int(sys.argv[1])]
    density_bins = [0,2,4,6,8,10,11,12,13,18]
    for i in np.arange(len(density_bins)):
        if density<density_bins[i]:
            break
        fakefilename = 'stellar_density_'+str(density_bins[i])+'_'+str(density_bins[i+1])+'.matchfake'
    
    # Files
    photfile = os.path.join(phot_dir, '{0:s}.match'.format(name))
    #point to correct matchfake file 
    fakefile = os.path.join(fake_dir,fakefilename)
    logfile = os.path.join(phot_dir, '{0:s}.xml'.format(name))
    # Check that paths exists where files are to be written
    for filename in [logfile]:
        safe_mkdir(os.path.dirname(filename))

    # Check if an xml log file exists and create one if not
    try:
        tree = etree.parse(logfile)
    except IOError:
        table_xml = etree.Element('TABLE')
        coldef_xml = etree.SubElement(table_xml, 'COLDEF')
        data_xml = etree.SubElement(table_xml, 'DATA')

        coldef_list = [('A_V', 'float'), ('dA_V', 'float'), ('fit', 'float')]
        for name, dtype in coldef_list:
            c_xml = etree.SubElement(coldef_xml, 'C')
            name_xml = etree.SubElement(c_xml, 'NAME')
            dtype_xml = etree.SubElement(c_xml, 'DTYPE')
            name_xml.text, dtype_xml.text = name, dtype

        format_table(table_xml)
        tree = etree.ElementTree(table_xml)
        tree.write(logfile, encoding='UTF-8', xml_declaration=True)

    # Get results from previous runs
    table_xml = tree.getroot()
    data_xml = table_xml[1]
    record = {(row[0].text, row[1].text): row[2].text for row in data_xml}
    #Default should be PADUA2006_CO_AGB (all caps)
    if sys.argv[3] == 'Padua2006_CO_AGB':
        models = None
    else:
        models = sys.argv[3]


    IMF = 'Kroupa'
    #dmod = (24.47, 24.47)
    #dmod = (24.67, 24.67) #ML Edit July 27, 2020 with deGrijs+2017 M33 distance modulus
    dmod = (29.4, 29.4) #Edit 20220530 with Murphy+2018 NGC 6946 TRGB distance modulus
    res = 0.05
    #logZ = (min logZ, max logZ, step logZ)
#     #  Padua2006_CO_AGB: Marigo et al. 2008, A&A, 482, 883 and
#                     Girardi et al. 2010, ApJ, 724, 1030
#     metallicity range: -2.3 <= [M/H] <= +0.2
#     age range: 4 Myr - 15.85 Gyr (6.6 < logt < 10.2)
#     mass range: 0.15 - 120 Msun
#     notes:
#      -the most recent Padua models include C/O ratio, and thus use different
#       transformations for C-stars as for oxygen-rich stars
#      -The common opionion is that the Padua models produce red giant branch
#       stars that are too blue.  This will induce MATCH to fit those
#       populations using somewhat higher metallicities.  While I haven't
#       actually observed this causing problems in the SFR vs. time
#       measurements (it will, of course, produce a spuriously low enrichment
#       rate), this could be mitigated by use of short+wide combination gates
#       if needed.


    if sys.argv[3] == 'Padua2006_CO_AGB':
        logZ = (-1, 0.2, 0.1)
    elif sys.argv[3] == 'MIST':
        logZ = (-1, 0.5, 0.1)
    elif sys.argv[3] == 'PARSEC':
        logZ = (-1, 0.6, 0.1)
    else:
        logZ = (-1, 0.2, 0.1) #use Padua range as default
    BF = 0.35
    badfrac = (1e-6, 1e-6)
    #zinc = (-2.3, -0.9, -1.4, 0.1)
    #Edit July 28, 2020
    #Edited to match description in paper: 
    #Range for oldest time bin should be -2.3 to -0.9
    #Range for youngest time bin should be -0.4 to 0.1
    zinc = (-1, -0.3, -0.4, 0.2)
    dAvy = 0
    
    #Edit July 28, 2020 - default is Padua2006_CO_AGB, so while no model is listed, that is the model being used
    #models = 'Padua2006_CO_AGB'

    filter1 = {'name': 'UVIS275W', 'min': 18.0, 'max': f275w_50_comp}
    filter2 = {'name': 'UVIS336W', 'min': 18.0, 'max': f336w_50_comp}
    CMD = {'V': filter1, 'I': filter2,
           'Vres': 0.1, 'V-I': (-1.3, 3.3, 0.05), 'fake_sm': 3}

    t = [6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0] #Edit 20220530: youngest possible time to 100Myr 
    tbins = list(zip(t[:-1], t[1:]))


    # Run calcsfh in a given opration mode
    a = sys.argv
    mode = sys.argv[4]
    if mode == 'pattern':
        Av0, dAv0 = float(a[5]), float(a[6])  # Initial seed point
        d_Av, d_dAv = float(a[7]), float(a[8])  # initial step size
        d_Av0, d_dAv0 = float(a[9]), float(a[10])  # Minimum step size
        #Avlim, dAvlim = (0.0, 2.5), (0.0, 1.5)  # Constrain Av and dAv
        #Edit August 10, 2020
        Avlim, dAvlim = (0.0, 1.5), (0.0, 2.5)  # Constrain Av and dAv
        pattern_search(func, (Av0, dAv0), (d_Av, d_dAv), (d_Av0, d_dAv0),
                       xlim=Avlim, ylim=dAvlim)


if __name__ == '__main__':
    main()