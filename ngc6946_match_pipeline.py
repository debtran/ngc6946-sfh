import os
import subprocess
import shutil
import sys
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from datetime import datetime
import time
import logging #use this to create time stamped log files for the pipeline process
def main():
    """
    To run this script:
    python m33_match_pipeline.py {match_dir} {proj_dir} {sfh_region} {model}
    Example (run from proj_dir):
    python m33_match_pipeline.py /gscratch/astro/mlazz/match2.7/bin/calcsfh /gscratch/astro/detran/MATCH/NGC6946/ 2265 Padua2006_CO_AGB
    
    

    
    Note on model selection:
    To compute systematic errors, several stellar models will be run: Padua, MIST, Dartmouth, Victoria.
    Models will need to be called by the handles listed below for them to work with MATCH
    Also: depending on MATCH installation, these stellar models must be downloaded and in the 
    match2.7/ directory and be unzipped for it to work.
    The models have unique flags in MATCH (case sensitive!):
    -Padua2006_CO_AGB #Padua models with C/O and AGB stars
    -Victoria_ss #Victoria models with solar metallicity
    -MIST #MIST models with TP-AGB
    -Dartmouth_afep0 #Dartmouth models with solar metallicity
    """
    #Path to MATCH directory that contains the calcsfh, hybridMC, zcombine, zcmerge commands
    #Example on UW astro network: /astro/apps7/opt/match2.7/bin/
    start_time = time.time()
    
    match_commands = sys.argv[1]
    proj_dir = sys.argv[2]
    region_id = sys.argv[3] #four digit string corresponding to region number: 0001, 0125, etc.
    ## If you want to use the default model, Padua2006_CO_AGB, do not list a model!
    if (len(sys.argv) == 4)|(sys.argv[4] == 'Padua2006_CO_AGB'): #If no input model, use MATCH default of PADUA2006_CO_AGB
        model = 'Padua2006_CO_AGB'
        model_flag = ' '
    else: #If model is given, use that model for all calcsfh runs
        model = sys.argv[4] #Stellar model used by MATCH, will need various for systematic comparison
        model_flag = '-'+model
    
    #Proj dir should contain the following:
    #m33_match_pipeline.py
    #m33_runcalcsfh.py
    
    
    
    #Make sure file structure is in place:
    #Check if a directory for the model being used exists, if not, create it:
    region_dir = proj_dir+'grids/NGC6946_'+region_id+'/' #Directory for the given SFH region, within model_dir
    archive_path = proj_dir+'NGC6946_'+region_id+'/archive/'
    
    if os.path.isdir(region_dir)==False:
        os.mkdir(region_dir)
    #Now make sure that the region dir has an arhive subdir:
    if os.path.isdir(archive_path)==False:
        os.mkdir(archive_path)
    
    # Create a log file for the whole pipeline that will record commands, values, etc.
    #If previous pipeline logfile exists, move it to a new filename so a new file can be created
    log_file_path = region_dir+'NGC6946_'+region_id+'_pipeline.log'
    logging.basicConfig(filename=log_file_path,
                        level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    #if os.path.isfile(log_file_path) == True:
    #    shutil.move(log_file_path,log_file_path+'_old')
    #log_file = open(log_file_path,'a')
    logging.info('Region '+region_id)
    
    #Add some details about the run:
    logging.info('Match installation used = '+match_commands)
    logging.info('Stellar model being used = '+model)
    
    ## Make sure AST file is in place: ##
    
    #AST files already exist- need to figure out how to translate current density stuff into this. 
    

    #Read in F275W and F336W completeness (determined by density bin previously) to define the magnitude limits to include on the CMD:
    #Edit 20220610, D. Tran 
    density_comp = pd.read_csv(proj_dir + '/grid_density_completeness.csv')
    f275w_50_comp = density_comp.F275_comp.values[int(sys.argv[1])] #completeness val of the specific grid
    f336w_50_comp = density_comp.F336_comp.values[int(sys.argv[1])]
    
    #point to correct fake stars based on density bin
    density = density_comp.denisty[int(sys.argv[1])]
    density_bins = [0,2,4,6,8,10,11,12,13,18]
    for i in np.arange(len(density_bins)):
        if density<density_bins[i]:
            break
        fakefilename = 'stellar_density_'+str(density_bins[i])+'_'+str(density_bins[i+1])+'.matchfake'
    
    #Define path of AST files for each stellar density bin
    ast_path = proj_dir+'fake/'
    #Define path that MATCH input AST file should have:
    ast_file = ast_path+fakefilename
    logging.info('AST file used = '+ast_file)
        
    ## Make sure photometry file is in place ##
    phot_file = region_dir +'NGC6946_'+region_id+'.match'
    
    ################ STEP 1 ################
    ## Run m33_runcalcsfh.py script to determine best fit Av and dAv ##
    #skip_runcalcsfh = 1 #if value = 1, skip the Av, dAv search and move on to Step 2
    runcalcsfh_path = '/gscratch/astro/detran/MATCH/ngc6946_runcalcsfh_hyak.py'
        
    #if skip_runcalcsfh == 0:
        
    #Set run calcsfh command
    #Final values correspond to:
    #Pattern search for Av/dAv
    #Seed value (starting point): Av=0.114, dAv=0.0
    #Initial step size in Av, dAv = 0.2, 0.2
    #Minimum step size in Av, dAv = 0.05, 0.05
    
    run_calcsfh_command = 'python '+runcalcsfh_path+' NGC6946_'+region_id+' '+region_dir+'/ '+model+ ' pattern 0.8 0.0 0.2 0.2 0.05 0.05'
    #Set off run_calcsfh run (this takes a while to run, a few hours on sunbreak):
    logging.info('run_calcsfh_command = '+run_calcsfh_command)
    runcalcsfh_complete = subprocess.run(run_calcsfh_command, shell=True, check=True)
    runcalcsfh_success = runcalcsfh_complete.returncode

    if runcalcsfh_success != 0:
        #Runcalc sfh exited with an error, print error message
        logging.info('Run calcsfh not successful')
    else:
        #Continue on to next step using runcalcsfh output
    #if skip_runcalcsfh == 1:
    
        ################ STEP 2 ################
        ## Read in XML file which outlines the parameter search and fit values for Av, dAv combinations: ##
        xml_file = region_dir+'/NGC6946_'+region_id+'.xml'
        line_count = 0
        with open(xml_file, 'r') as f:
            for line in f:
                if line.startswith('    <R>'):
                    line_count += 1
                    line_text = line.split('>')
                    fit_string = line_text[6]
                    fit = float(fit_string[:-3])
                    av_string = line_text[2]
                    av = float(av_string[:-3])
                    dav_string = line_text[4]
                    dav = float(dav_string[:-3])
                    if line_count == 1:
                        current_min = fit
                        best_av = av #Edit Feb 22, 2021: not defining initial av, dav as best results in bug when the first AV/DAV combination has the lowest fit value
                        best_dav = dav
                    else:
                        if fit < current_min:
                            current_min = fit
                            best_av = av
                            best_dav = dav
        logging.info('Best fit Av, dAv, fit value = '+str(best_av)+','+str(best_dav)+','+str(current_min))
        
        
        ################ STEP 3 ################
        ## Write new param file with best fit Av values: ##
        ## Also edit to include the max F475W mag and F814W mag from the 50% completeness limits
        
        #Set name of new best fit param file
        #Set path to the /archive/ directory
        #best_fit_param_file = 'M33-'+region_id+'_best_fit_Av-dAv_'+str(best_av)+'-'+str(best_dav)+'.param'
        best_fit_param_file = 'NGC-'+region_id+'.param'
        

        #Copy the template param file for M33:
        shutil.copyfile(proj_dir+'NGC6946_calcsfh_param_'+model+'.txt',archive_path+best_fit_param_file)

        #Read in the copy of the template param file
        param_file = open(archive_path+best_fit_param_file, "r")
        param_file_lines = param_file.readlines()
        #Change the first line to reflect the best fit Av values (dAv doesn't appear in the param file, it is fed in 
        #to calcsfh in the command)
        param_file_lines[0] = "  29.4 29.4 0.05 "+str(best_av)+" "+str(best_av)+" 0.05\n"
        #Change the lines indicating the magnitude limits to reflect the correct limits for this stellar density bin
            param_file_lines[5] = "18.00 "+str(f275w_50_comp)+"0 UVIS275W\n"
        param_file_lines[6] = "18.00 "+str(f336w_50_comp)+"0 UVIS336
        W\n"
        #Open the param file that was copied and change the first line to reflect the best fit Av
        param_file = open(archive_path+best_fit_param_file, "w+")
        param_file.writelines(param_file_lines)
        param_file.close()
        
        ################ STEP 4 ################
        ## Re-run calcsfh with the best fit Av and dAv values ##
        calcsfh_path = match_commands+'calcsfh'
        param_file = archive_path+best_fit_param_file
        phot_file = region_dir+'/NGC6946_'+region_id+'.match'
        ast_file = ast_path+fakefilename
        output_sfh = archive_path+'NGC6946_'+region_id+'.sfh'
        calcsfh_log_file = archive_path+'NGC6946-'+region_id+'.log'


        #Construct calcsfh command with -mcdata flag, and best fit dAv set as a flag (-dAvy is always 0.00)
        calcsfh_mcdata_command = calcsfh_path+' '+param_file+' '+phot_file+' '+ast_file+' '+output_sfh+' -zinc -dAv='+str(best_dav)+' -dAvy=0.00 -kroupa -mcdata '+model_flag+' > '+calcsfh_log_file
        logging.info('calcsfh -mcdata command ='+calcsfh_mcdata_command)

        #Only proceed if the paths to the files work:
        if os.path.isfile(param_file)&os.path.isfile(phot_file)&os.path.isfile(ast_file):
            calcsfh_mcdata_complete = subprocess.run(calcsfh_mcdata_command, shell=True, check=True)
            calcsfh_mcdata_success = calcsfh_mcdata_complete.returncode
        else:
            logging.info('Cannot execute calcsfh -mcdata command as written:')
            logging.info(calcsfh_mcdata_command)
            calcsfh_mcdata_success = -999
        
        if calcsfh_mcdata_success != 0:
            logging.info('mcdata calcsfh command failure')
        else:
            #Write the number of real and fake stars fit out to log file
            with open(calcsfh_log_file) as f:
                datafile = f.readlines()
            for line in datafile:
                if 'real stars read' in line:
                    # found = True # Not necessary
                    logging.info(line)
                if 'fake stars read' in line:
                    logging.info(line)
            #Continue!
            
            ################ STEP 5 ################
            ## Determine optimal hybridMC parameters: 
            #DT Edit 20220621: Using 0.2 and 0.4 for dt and tint respectively 
            #based on 15 regions of varying stellar density
            dt = 0.2
            tint = 0.4
            logging.info('hybridMC dt, tint values used ='+str(dt)+','+str(tint))
            
            
            #Generate hybridMC command using best dt and tint values:
            hybridMC_path = match_commands+'hybridMC'
            hybridMC_sfh_input = archive_path+'NGC6946_'+region_id+'.sfh.dat'
            hybridMC_out = archive_path+'NGC6946_'+region_id+'.out.mcmc'
            hybridMC_log = archive_path+'NGC6946_'+region_id+'.hybridMC.log'
            nmc=10000
            hybridMC_command = hybridMC_path+' '+hybridMC_sfh_input+' '+hybridMC_out+' -nmc='+str(nmc)+' -nsfr=0 -dt='+str(dt)+' -tint='+str(tint)+' > '+hybridMC_log
            logging.info('hybridMC command ='+hybridMC_command)
            
            #Make sure sfh.dat file exists before running:
            if os.path.isfile(hybridMC_sfh_input):
                hybridMC_complete = subprocess.run(hybridMC_command, shell=True, check=True)
                hybridMC_success = hybridMC_complete.returncode
            else:
                logging.info('Cannot execute hybridMC command as written: ')
                logging.info(hybridMC_command)
                hybridMC_success = -999

            if hybridMC_success != 0:
                logging.info('hybridMC command failure')
                logging.info(hybridMC_complete)
            else:
                #Continue!
                
                ################ STEP 6 ################
                ## Use `zcombine` to turn the best fit .sfh file into a .sfh.zc file that can be merged with the errors:
                zcombine_path = match_commands + 'zcombine'
                sfh_file = archive_path+'NGC6946_'+region_id+'.sfh'
                sfh_zc_file = archive_path+'NGC6946_'+region_id+'.sfh.zc'

                zcombine_command = zcombine_path+' '+sfh_file+' > '+sfh_zc_file
                logging.info('zcombine command ='+zcombine_command)
                
                if os.path.isfile(sfh_file):
                    zcombine_complete = subprocess.run(zcombine_command, shell=True, check=True)
                    zcombine_success = zcombine_complete.returncode
                else:
                    logging.info('Cannot execute zcombine command as written:')
                    logging.info(zcombine_command)
                    zcombine_success = -999
                
                if zcombine_success != 0:
                    logging.info('zcombine command failure')
                else:
                    #Continue!
                    
                    ################ STEP 7 ################
                    ## Use `zcombine` again to merge the sfh.zc file with the out.mcmc file 
                    ##  to create an out.mcmc.zc file with random errors included:
                    
                    out_mcmc_file = archive_path+'NGC6946_'+region_id+'.out.mcmc'
                    out_mcmc_zc_file = archive_path+'NGC6946_'+region_id+'.out.mcmc.zc'

                    zcombine_command_2 = zcombine_path+' -unweighted -medbest -jeffreys -best='+sfh_zc_file+' '+out_mcmc_file+' > '+out_mcmc_zc_file
                    logging.info('zcombine command 2='+zcombine_command_2)
                    
                    #Make sure two input files exist before sending off the zcombine command:
                    if os.path.isfile(sfh_file):
                        zcombine_2_complete = subprocess.run(zcombine_command_2,shell=True,check=True)
                        zcombine_2_success = zcombine_2_complete.returncode
                    else:
                        logging.info('Cannot execute zcombine 2 command as written: ')
                        logging.info(zcombine_command_2)
                        zcombine_2_success = -999

                    if zcombine_2_success != 0:
                        logging.info('zcombine 2 command failure')
                    else:
                        #Continue!
                        
                        ################ STEP 8 ################
                        ## Use `zcmerge` to merge the sfh.zc and the out.mcmc.zc files to create a readable SFH file
                            
                        zcmerge_path = match_commands+'zcmerge'
                        final_sfh_file = archive_path+'NGC6946_'+region_id+'.final.txt'

                        zcmerge_command = zcmerge_path+' '+sfh_zc_file+' '+out_mcmc_zc_file+' -absolute > '+final_sfh_file
                        logging.info('zcmerge command ='+zcmerge_command+'\n')
                        if os.path.isfile(sfh_zc_file)&os.path.isfile(out_mcmc_zc_file):
                            zcmerge_complete = subprocess.run(zcmerge_command,shell=True,check=True)
                            zcmerge_success = zcmerge_complete.returncode
                        else:
                            logging.info('Cannot execute zcmerge command as written:')
                            logging.info(zcmerge_command)
                            
                        if zcmerge_success != 0:
                            logging.info('zcmerge command failure')
                        else:
                            logging.info('Pipeline complete for region '+region_id)
                            logging.info('Final SFH saved:')
                            logging.info(final_sfh_file)
                            run_time = time.time() - start_time
                            logging.info('Total runtime = '+str(run_time)+' seconds')
                            
                            #Clean up some intermediate files:
                            # Don't need to hang onto any of the intermediate .sfh and .sfh.cmd files
                            # generated by runcalcsfh
                            extra_files_cleanup = 'rm '+archive_path+'[0,1].*'
                            subprocess.run(extra_files_cleanup,shell=True)

                            # Compress out.mcmc files and remove the original because they are huge!
                            compress_out_mcmc = 'tar cfz '+out_mcmc_file+'.tar.gz '+out_mcmc_file
                            rm_out_mcmc = 'rm '+out_mcmc_file
                            subprocess.run(compress_out_mcmc,shell=True)
                            subprocess.run(rm_out_mcmc,shell=True)
                            logging.info('cleanup complete, mcmc file compressed')

if __name__ == '__main__':
    main()
