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
import importlib

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
clogin  = 'dgu041'                               # login name
chost   = 'saga.sigma2.no'                       # login host
cwpath  = 'zemba/work/'+prefix                  # work path 
chpath  = 'zemba/model/'                        # model path 
copath  = '/cluster/work/users/dgu041/'+prefix   # run path 


# =====
# Paths
# =====

current_dir = "/mnt/c/Users/Daniel/'OneDrive - University of Bergen'/zemba_v1.1/phd_paper2"
script_dir  = "/mnt/c/Users/Daniel/'OneDrive - University of Bergen'/zemba_v1.1"
input_dir   = "/mnt/c/Users/Daniel/'OneDrive - University of Bergen'/zemba_v1.1/input"

# ===================
# List of input files
# ===================

kyr_init  = 1850
kyr_end   = 1200

input_list = [  

                # standard run
                # 'input_transient_standard',
                
                # lgm run
                # 'input_transient_lgm',
                
                # CO2 trend
                #'input_transient_co2_trend',
                
                # pre-MPT glacial
                #'input_transient_preMPT',
                #'input_transient_preMPTgl',
                #'input_transient_pi',
                #'input_transient_LGM',
                #'input_transient_preMPT_obliquity',
                #'input_transient_preMPT_precession',
                #'input_transient_preMPT_fxoa',
               # 'input_transient_preMPT_fxla',
                #'input_transient_preMPT_fxloa',
                'input_transient_preMPT_fxloasi',
                #'input_transient_preMPT_fxsi',
                #'input_transient_preMPT_fxsc',
                #'input_transient_preMPT_fxsic',
                #'input_transient_preMPT_obliquity_fxoa',
                #'input_transient_preMPT_obliquity_fxla',
                #'input_transient_preMPT_obliquity_fxloa',
                'input_transient_preMPT_obliquity_fxloasi',
                #'input_transient_preMPT_obliquity_fxsi',
                #'input_transient_preMPT_obliquity_fxsc',
                #'input_transient_preMPT_obliquity_fxsic',
                #'input_transient_preMPT_precession_fxoa',
                #'input_transient_preMPT_precession_fxla',
                #'input_transient_preMPT_precession_fxloa',
                'input_transient_preMPT_precession_fxloasi',
                #'input_transient_preMPT_precession_fxsi',
                #'input_transient_preMPT_precession_fxsc',
                #'input_transient_preMPT_precession_fxsic',
                
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
    
    ssh = subprocess.Popen(["wsl", "sftp", "saga"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            shell=True,
                            bufsize=0)
    
    commands = f"""
    put -r {script_dir}/input/other_data {chpath}
    put {script_dir}/albedo_core.py {chpath}
    put {script_dir}/atm_transport_core.py {chpath}
    put {script_dir}/hydrology_core.py {chpath}
    put {script_dir}/initialize_zemba.py {chpath}
    put {script_dir}/laskar_2006.txt {chpath}
    put {script_dir}/ocean_transport_core.py {chpath}
    put {script_dir}/radiation_core.py {chpath}
    put {script_dir}/seaice_core.py {chpath}
    put {script_dir}/snow_core.py {chpath}
    put {script_dir}/solar_forcing.py {chpath}
    put {script_dir}/zemba.py {chpath}
    put {script_dir}/input/sc_fraction_pi_res1.0.txt {chpath}
    put {script_dir}/input/si_fraction_pi_res1.0.txt {chpath}
    put {script_dir}/input/si_temperature_pi_res1.0.txt {chpath}
    put {script_dir}/input/lalbedo_pi_res1.0.txt {chpath}
    put {script_dir}/input/oalbedo_pi_res1.0.txt {chpath}
    bye
    """
        
    stdout, stderr = ssh.communicate(commands)  # Ensures all commands are executed
    ssh.wait()  # Ensure process completes before continuing
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    ssh.kill()  # Ensure process completes before continuing
    

    # =====================
    # Launch transient runs
    # =====================
    
    for sim in input_list: # for each experiment...
    
        try:
            
            # edit .ssh file
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v1.1/phd_paper2/drafts/{prefix}.ssh', 3, 'NAME=%s\n' % (sim))
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v1.1/phd_paper2/drafts/{prefix}.ssh', 53,'cp ${SRC}/%s.py %s.py\n' % (sim, sim))
           
            # edit transient_run.py
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v1.1/phd_paper2/drafts/run_orbital.py', 36, "input_file_name = '{0}'\n".format(sim))
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v1.1/phd_paper2/drafts/run_orbital.py', 38, "kyr_init = {0}\n".format(kyr_init))
            replace_line(f'C:/Users/Daniel/OneDrive - University of Bergen/zemba_v1.1/phd_paper2/drafts/run_orbital.py', 39, "kyr_end  = {0}\n".format(kyr_end))
            
            
            # make sub-folder in hpc cluster
            ssh = subprocess.Popen(["wsl", "sftp", "saga"],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    shell=True,
                                    bufsize=0)
            
            commands = f"""
            cd /cluster/home/dgu041/{cwpath}
            mkdir {sim}
            cd /cluster/home/dgu041/
            put {current_dir}/drafts/{prefix}.run {cwpath}/{sim}
            put {current_dir}/drafts/{prefix}.ssh {cwpath}/{sim}
            put {current_dir}/drafts/run_orbital.py {cwpath}/{sim}
            put {input_dir}/{sim}.py {cwpath}/{sim}
            bye
            """
            
            stdout, stderr = ssh.communicate(commands)  # Ensures all commands are executed
            ssh.wait()  # Ensure process completes before continuing
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            ssh.kill()
            

            
           
            # launch model run
            
            ssh = subprocess.Popen(["wsl", "ssh" , "-T", "saga"],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    shell=True,
                                    bufsize=0)
            
            ssh.stdin.write(f"cd /cluster/home/dgu041/{cwpath}/{sim} && dos2unix {prefix}.ssh && chmod u=wrx {prefix}.ssh && ./{prefix}.ssh")
            ssh.stdin.close()
            for line in ssh.stderr:
                print(line.strip())
            for line in ssh.stdout:
                print(line.strip())
            ssh.kill()
            
            
            
            

            
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
        
        # name
        name    = sim.replace('input_', '')   # 'input_transient_preMPT' -> 'transient_preMPT'
        version = 'moist'
        res     = '1.0'

        
        # retrieve
        try: 
            ssh = subprocess.Popen(["wsl", "sftp", "saga"],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    shell=True,
                                    bufsize=0)
            
            commands = f"""
            cd {copath}/{sim}/
            get output_{name}_{version}_res{res}_meta.pkl {current_dir}/{sim}/
            get output_{name}_{version}_res{res}.h5 {current_dir}/{sim}/
            """
            
            stdout, stderr = ssh.communicate(commands)  # Ensures all commands are executed
            ssh.wait()  # Ensure process completes before continuing
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            ssh.kill()  # Ensure process completes before continuing
            
            
    
    
        except:
            print('Failed to copy over file.')
            
        
    
    