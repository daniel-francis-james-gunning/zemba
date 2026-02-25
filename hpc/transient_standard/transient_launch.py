# -*- coding: utf-8 -*-
"""
Script for launching transient simulations of ZEMBA to HPC...
"""

# import packages
import numba    as nb 
from   numba    import njit
import numpy    as np
from   pathlib  import Path
import os
import shutil
import subprocess
import sys

# function to edit/replace file lines
def replace_line(file_name, line_number, text):
    lines = open(file_name, 'r').readlines() 
    lines[line_number] = text                   
    out = open(file_name,'w')
    out.writelines(lines)                    
    out.close()
    
# ===========================
# Send or retreive simulation
# ===========================
    
spfc = input("Send out (s) or retrieve (r) runs? (s/r): ")

if spfc != 's' and spfc != 'r':
    print('Needs to be s or r.')
    sys.exit()

# ==========================
# Specification for saga run
# ==========================

# prefix
prefix  = 'transient_standard'

# hpc cluster specifications
clogin  = 'dgu041'                               # cluster login name
chost   = 'saga.sigma2.no'                       # cluster login host
cwpath  = '~/zemba/work/'+prefix                 # cluster work path 
chpath  = '~/zemba/model/'                       # cluster model path 
copath  = '/cluster/work/users/dgu041/'+prefix   # cluster run path 

# =====
# Paths
# =====

current_dir = 'C:/Users/Daniel/"OneDrive - University of Bergen"/zemba_v01/hpc/transient_standard'
script_dir  = 'C:/Users/Daniel/"OneDrive - University of Bergen"/zemba_v01/'
input_dir   = 'C:/Users/Daniel/"OneDrive - University of Bergen"/zemba_v01/input'


# ===================
# List of input files
# ===================

input_list = [  

                # standard run
                # 'input_transient_standard',
                
                # lgm run
                # 'input_transient_lgm',
                
                # CO2 trend
                'input_transient_co2_trend'
                
                # pre-MPT glacial
                #'input_transient_preMPTgl'
                
                # dry run
                #'input_transient_dry',
                
                # only obliquity (fixed precession)
                #'input_transient_obliquity',
                
                # only precession (fixed obliquity)
                #'input_transient_precession',
                
                # prescribed ice cover changes
                #'input_transient_ice',
                
                # prescribed co2 changes
                #'input_transient_co2',
                
                # prescribed ice cover and co2 changes 
                 #'input_transient_ice_co2',
                
                # 180 ppm CO2 run
                #'input_transient_180ppm',
                
                # 380 ppm CO2 run
                #'input_transient_380ppm',
                
                # fixed ocean albedo
                #'input_transient_fxoa',
                
                # fixed land albedo
                #'input_transient_fxla',
                
                # fixed ocean and land albedo
                #'input_transient_fxoa_fxla',
              ]

if spfc == 's':

    # ========
    # Run file
    # ========
    
    
    # check number of simulations ... 
    num_simulations = len(input_list)
    print("Number of simulations = " + str(num_simulations) + "\n")
    
    # prompt user to continue
    response = input("Do you want to continue? (y/n): ")
    if (response == "y"):
        print("\n\nproceeding...")  
    else:
        sys.exit()
    
    # ========================
    # SCP files to remote host
    # ========================
    
    subprocess.call(f'scp -p {script_dir}/zemba.py {clogin}@{chost}:{chpath}')             # zemba main file
    subprocess.call(f'scp -p {script_dir}/utilities.py {clogin}@{chost}:{chpath}')         # zemba utility functions file
    subprocess.call(f'scp -p {script_dir}/initialize_zemba.py {clogin}@{chost}:{chpath}')  # zemba initialize functions file
    subprocess.call(f'scp -p {script_dir}/solar_forcing.py {clogin}@{chost}:{chpath}')     # solar forcing file
    subprocess.call(f'scp -p {script_dir}/laskar_2006.txt {clogin}@{chost}:{chpath}')      # laskar solution text file
    subprocess.call(f'scp -r {script_dir}/input/other_data {clogin}@{chost}:{chpath}')     # other data folder
    subprocess.call(f'scp -r {script_dir}/input/other_data {clogin}@{chost}:{chpath}')     # other data folder
    subprocess.call(f'scp -r {script_dir}/CO2.txt {clogin}@{chost}:{chpath}')              # other data folder
    subprocess.call(f'scp -r {script_dir}/ice_fraction.txt {clogin}@{chost}:{chpath}')     # other data folder
    subprocess.call(f'scp -r {script_dir}/pi_land_albedo.txt {clogin}@{chost}:{chpath}')   # pre-industrial land albedo
    subprocess.call(f'scp -r {script_dir}/pi_ocean_albedo.txt {clogin}@{chost}:{chpath}')  # pre-industrial ocean albedo
    subprocess.call(f'scp -r {script_dir}/si_fraction.txt {clogin}@{chost}:{chpath}')      # pre-industrial land albedo
    subprocess.call(f'scp -r {script_dir}/si_temperature.txt {clogin}@{chost}:{chpath}')  # pre-industrial ocean albedo
    subprocess.call(f'scp -r {script_dir}/sc_fraction.txt {clogin}@{chost}:{chpath}')     # pre-industrial ocean albedo
    
    # =====================
    # Launch transient runs
    # =====================
    
    for sim in input_list: # for each experiment...
    
        try:
            
            # edit .ssh file
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v01/hpc/{prefix}/drafts/{prefix}.ssh', 3, 'NAME=%s\n' % (sim))
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v01/hpc/{prefix}/drafts/{prefix}.ssh', 43,'cp ${SRC}/%s.py %s.py\n' % (sim, sim))
           
            # edit transient_run.py
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v01/hpc/{prefix}/drafts/transient_run.py', 28, "name_of_input_file = '{0}'\n".format(sim))
            
            # make sub-folder in hpc cluster
            ssh = subprocess.Popen(["ssh", "-T", "dgu041@saga.sigma2.no"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,universal_newlines=True,bufsize=0, shell=True)
            ssh.stdin.write(f'cd {cwpath} && mkdir {sim}')    
            stdout, stderr = ssh.communicate()
            print("Completed run for {} threads".format(sim))
            
            # copy over files to host...
            subprocess.call(f'scp -p {current_dir}/drafts/{prefix}.run {clogin}@{chost}:{cwpath}/{sim}')       # .run file
            subprocess.call(f'scp -p {input_dir}/{sim}.py {clogin}@{chost}:{cwpath}/{sim}')                    # input.py file
            subprocess.call(f'scp -p {current_dir}/drafts/{prefix}.ssh {clogin}@{chost}:{cwpath}/{sim}')       # .ssh file
            subprocess.call(f'scp -p {current_dir}/drafts/transient_run.py {clogin}@{chost}:{cwpath}/{sim}')   # transient_run.py
            
            # launch model run
            ssh = subprocess.Popen(["ssh","-T", f"{clogin}@{chost}"],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    shell=True,
                                    bufsize=0)
            
            ssh.stdin.write(f"cd {cwpath}/{sim} && dos2unix {prefix}.ssh &&chmod u=wrx {prefix}.ssh && ./{prefix}.ssh")
            ssh.stdin.close()
            for line in ssh.stderr:
                print(line.strip())
            for line in ssh.stdout:
                print(line.strip())
            
        except: 
            
            print('d')
        
        
        
elif spfc == 'r':

    # ========================
    # Retreieve transient runs
    # ========================
    
    for sim in input_list: # for each experiment...
    
        # make dir for results
        try:
            os.mkdir(f'{os.getcwd()}/{sim}')
        
        except:
            
            pass
    
        
        try: # retreive relevant files...
            subprocess.call('scp -r %s@%s:%s/%s/%s_moist_res5.0.pkl %s/%s/' % (clogin, chost, copath, sim, sim.replace('input_','output_'), current_dir, sim)) # output file
        except:
            print('Failed to copy over file.')
        
    
    