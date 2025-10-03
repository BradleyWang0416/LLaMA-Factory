import os
import numpy as np

def save_numpy_files(source_data_dict, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key in source_data_dict:
        if key in ['slice_id', 'image_sources']:
            continue
        save_path = os.path.join(save_dir, key)
        if not os.path.exists(save_path): 
            os.makedirs(save_path)

        for sample_id in range(len(source_data_dict[key])):
            name_suffix = f"slice{int(source_data_dict['slice_id'][sample_id][0])}_{int(source_data_dict['slice_id'][sample_id][-1]+1)}"
            save_file = os.path.join(save_path, f"h36m_{sample_id:06d}_{name_suffix}.npy")
            np.save(save_file, source_data_dict[key][sample_id])