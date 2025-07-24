# **3D Timeseries Data Subsampling Script**

This Python script is designed to preprocess and subsample large 3D timeseries datasets generated from Fortran-based simulations. It reads a series of binary files, performs spatial and channel-wise subsampling, and packages the result into a single, compressed NumPy (.npz) archive, ready for machine learning or analysis workflows.

The script is optimized for very large datasets (terabyte-scale) by using memory-mapped files to keep RAM usage low and constant. It also supports parallel processing to significantly speed up execution on multi-core machines.

## **Features**

* **Memory Efficient:** Processes terabyte-scale data with minimal RAM usage via memory-mapped arrays.  
* **Parallel Processing:** Utilizes multiple CPU cores to process files in parallel for maximum speed.  
* **Flexible Subsampling:** Allows for even subsampling on the X-axis and specific index selection on the Y-axis.  
* **Channel Selection:** Manually specify which data channels (e.g., vx, vy, T) to include in the final output.  
* **Data Type Conversion:** Converts data from float64 (double) to float32 (single) to reduce final file size.  
* **Robust Verification:** Includes multiple checks to ensure data integrity:  
  * Verifies that the actual file size matches the size expected from metadata.  
  * Checks for time-invariant boundary conditions on the Z-axis.  
* **Mock Data Generation:** Includes a feature to generate a complete mock dataset for local testing and validation.

## **1\. Setup**

Before running the script, you need to install the required Python packages.

**Dependencies:**

* numpy  
* tqdm

Create a file named requirements.txt with the following content:

numpy  
tqdm

Then, install the dependencies using pip:

pip install \-r requirements.txt

## **2\. Input Data Format**

The script makes specific assumptions about the input data format. Please ensure your data matches this structure.

### **a. Grid Dimensions**

You must provide the grid dimensions in one of two ways:

1. **Via a metadata file** (--meta-file-path): A text file where dimensions represent the number of **spaces** between grid points.  
2. **Via direct command-line arguments** (--nx, \--ny, \--nz): The integer number of grid **points**.

**Example runParameters.txt:**

nX: 2300  
nY: 480  
nZ: 120  
... (other parameters are ignored)

### **b. Binary Timeseries Files**

* Each timestep is a separate binary file.  
* Filenames must follow the pattern \<prefix\>\<time\_index\>, where the time index is a zero-padded 6-digit number (e.g., patt3d\_vx3d\_000000, patt3d\_vx3d\_000001).  
* The file has **no EOF or EOL characters**.  
* The data is stored in the following order:  
  1. **X-Coordinates:** An array of Nx float64 values.  
  2. **Y-Coordinates:** An array of Ny float64 values.  
  3. **Z-Coordinates:** An array of Nz float64 values.  
  4. **Channel Data:** The main data block, organized as follows:  
     * The data is looped **by Z-index first**.  
     * For each Z-slice, the data is looped **by channel**.  
     * For each channel within that Z-slice, an (Nx, Ny) block of float64 values is written in **Fortran (column-major) order**.

## **3\. How to Run**

All configuration is handled via command-line arguments.

### **Full Command Structure**

python preprocess\_dns\_output.py \[ARGUMENTS\]

### **Key Arguments**

* **Dimension Source (provide one method):**  
  * \--meta-file-path: Full path to your runParameters.txt file.  
  * \--nx, \--ny, \--nz: The number of grid **points** for each axis.  
* \--input-dir: Path to the directory containing the raw binary files.  
* \--output-file: Path where the final .npz archive will be saved.  
* \--time-start, \--time-end: The range of time indices to process (e.g., \--time-start 0 \--time-end 2000).  
* \--num-workers: Number of parallel CPU cores to use. **Defaults to 1\.** Set to the number of cores allocated for your job for best performance. Set to \-1 to use all available cores on the machine.

### **Subsampling Arguments**

* \--num-x-samples: The number of evenly spaced points to select along the X-axis.  
* \--y-indices-to-keep: A space-separated list of the specific Y-indices you want to extract (e.g., \--y-indices-to-keep 10 50 100).  
* \--source-channel-labels: A space-separated list of **all** channel names in the exact order they appear in the binary files (e.g., \--source-channel-labels vx vy vz T).  
* \--channel-indices-to-keep: A space-separated list of the **indices** of the channels you want to save in the final file (e.g., \--channel-indices-to-keep 0 3 to keep vx and T).

### **Example Command**

This command processes timesteps 0 through 1999, using 16 CPU cores. It provides the dimensions directly, subsamples to 24 points on the X-axis, selects 3 specific Y-indices, and keeps all 4 channels.

python preprocess\_dns\_output.py \\  
    \--input-dir /path/to/raw\_data \\  
    \--output-file /path/to/output/subsampled\_data.npz \\  
    \--nx 2301 \--ny 481 \--nz 121 \\  
    \--time-start 0 \\  
    \--time-end 2000 \\  
    \--num-workers 16 \\  
    \--source-channel-labels vx vy vz T \\  
    \--channel-indices-to-keep 0 1 2 3 \\  
    \--num-x-samples 24 \\  
    \--y-indices-to-keep 10 55 80

## **4\. Important Recommendations**

### **Perform a Dry Run First\!**

Before launching a large job, **it is highly recommended to run the script on a small number of files first.** This will validate your setup and data assumptions.

\# Run on just the first two timesteps  
python preprocess\_dns\_output.py \--time-end 2 \[your other arguments\]

During this dry run, carefully check the output for:

* **File Size Warnings:** If you see a "FILE SIZE MISMATCH" warning, it means your specified dimensions do not match the content of your binary files.  
* **Boundary Check SUCCESS:** The script checks if the Z-boundaries are constant over time. If this check fails, it is a strong indicator that the assumed data layout (z \-\> channel \-\> Fortran(x,y)) is incorrect.

### **Output File Structure**

The final .npz file contains the following arrays (keys):

* timeseries: The 5D NumPy array of subsampled data with shape (Time, X, Y, Z, Channels) and float32 data type.  
* labels: A list of the channel names that were kept.  
* x\_coords, y\_coords, z\_coords: The coordinate values for the subsampled grid points.

You can load and inspect this file in Python as follows:

import numpy as np

data \= np.load('subsampled\_data.npz')

print(data.files)  
\# Output: \['timeseries', 'labels', 'x\_coords', 'y\_coords', 'z\_coords'\]

timeseries\_data \= data\['timeseries'\]  
print(timeseries\_data.shape)

