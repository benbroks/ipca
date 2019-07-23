import numpy as np
import sys
import os
import h5py

from sklearn.decomposition import IncrementalPCA

# PLACEHOLDER FUNCTION
def batch_array(rows,columns):
    return np.random.rand(rows,columns)

def incremental_dense(batch_rows=1000,total_rows=5000,columns=1000,save_fp="dense.h5"):
    # OVERWRITING FILE (IF IT EXISTS)
    if os.path.exists(save_fp):
        os.remove(save_fp)

    # USE APPEND MODE TO INCREMENTALLY INCREASE ARRAY SIZE 
    f = h5py.File(save_fp,mode="a")

    # USING FLOAT NP ARRAY (NOT A REQUIREMENT)
    dt = np.dtype(float)
    dset = f.create_dataset('float',(total_rows,columns),dtype=dt)

    for i in range(int(total_rows/batch_rows)):
        to_add = batch_array(batch_rows,columns)
        dset[i*batch_rows:(i+1)*batch_rows] = to_add
        # MINIMIZE RAM USAGE
        del to_add
    f.close()

def reading_batches(batch_rows=500,compressed_columns=200,uncompressed_fp="dense.h5",compressed_fp="compressed_dense.h5"):
    data = h5py.File(uncompressed_fp, 'r')
    total_rows = data["float"].shape[0]

    # TRAINING PCA MODEL INCREMENTALLY
    ipca = IncrementalPCA(n_components=compressed_columns)
    for i in range(int(total_rows/batch_rows)+1):
        pca_batch = data["float"][i*batch_rows:(i+1)*batch_rows]
        if len(pca_batch) == 0:
            break
        ipca.partial_fit(pca_batch)
    print("Partially fit.")

    # OUTPUT FILE
    if os.path.exists(compressed_fp):
        os.remove(compressed_fp)
    output = h5py.File(compressed_fp,mode="a")
    dt = np.dtype(float)
    dset = output.create_dataset('float',(total_rows,compressed_columns),dtype=dt)
    
    ## PERFORMING TRANSFORMATION/COMPRESSION
    for i in range(int(total_rows/batch_rows)+1):
        if i*batch_rows >= len(data["float"]):
            break
        pca_batch = ipca.transform(data["float"][i*batch_rows:(i+1)*batch_rows])
        dset[i*batch_rows:(i+1)*batch_rows] = pca_batch
        del pca_batch
    
    data.close()
    output.close()
    
if __name__ == "__main__":
    incremental_dense()
    print("Data set created.")
    reading_batches()
    print("Batches read.")
