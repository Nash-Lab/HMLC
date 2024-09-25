import numpy as np

s_0 = 'conda run --no-capture-output -n Process4DataAnalysis python3 -u ./04_ML.py'

with open('./SLURM_list_comands.txt', "w") as text_file:
    text_file.write('')

#%%

with open('./SLURM_list_comands.txt', "a") as text_file:
    for s in ['bin']:#['uni','bin','exp']:
        for ss in ['2D']:            
            for i,_ in enumerate(np.ones((501,501))[::10,:][:,::10].flatten()):
                for ii in range(10):
                    for iii in range(10):
                        text_file.write(s_0 + f' -s ../Results/bin_2D/ --coordinates_step 10 --n_lc 1  --n_mut_train_max 3 -c {i} --n_rep_shuffle {ii} --n_rep_error_calc {iii}\n')
            
    
#%%

# with open('./SLURM_list_comands.txt', "a") as text_file:
#     for s in ['exp']:#['uni','bin','exp']:
#         for ss in ['n_crit', 'h_crit', 'contour_crit_equidistant', 'eq_equidistant']:            
#             for i,_ in enumerate(range(51)):
#                 for ii in range(10):
#                     for iii in range(10):
#                         text_file.write(s_0 + f' -d {s} --mode_points {ss} --coordinates_step 10 --n_lc 10  --n_mut_train_max 3 -c {i} --n_rep_shuffle {ii} --n_rep_error_calc {iii}\n')


#%%

