### Installation and Usage: OpenMM with Martini 3

The SOM systems converted into the Martini 3 representation can be simulated using the **OpenMM** engine. This is made possible by the `martini_openmm` package, which converts GROMACS input files into OpenMM topologies.

Below is a workflow to set up the necessary Conda environment and run a simulation.

#### 1. Create the Environment
Create a new Conda environment containing Python and the OpenMM libraries:

```bash
conda create -n openmm python=3.12
conda activate openmm
conda install -c conda-forge openmm mdtraj
```

#### 2. GPU Support (Optional)
To run simulations on a GPU, you must install the CUDA toolkit compatible with your system's NVIDIA driver.
*Please check your driver version with `nvidia-smi`.*

For example, to install CUDA 12.5:
```bash
conda install -c conda-forge cuda=12.5
```

#### 3. Install Helper Packages
Install the required tools using `pip`.
*> **Note:** It is recommended to use `python -m pip` to ensure the packages are installed specifically into the active Conda environment.*

```bash
python -m pip install git+https://github.com/maccallumlab/martini_openmm.git
python -m pip install som2cg
```

#### 4. Prepare the Simulation Data
1.  **Generate Topologies:** Use `som2cg` to create the coarse-grained topology files. These will be stored in an `INIT_CG` folder by default.
2.  **Get Force Field Parameters:** Download the Martini 3 parameters and place them in the working directory.

    **Option A (Command Line - Linux/macOS):**
    ```bash
    wget https://cgmartini-library.s3.ca-central-1.amazonaws.com/1_Downloads/ff_parameters/martini3/martini_v300.zip
    unzip martini_v300.zip
    ```

    **Option B (Manual):**
    Download `martini_v300.zip` from [cgmartini.nl](https://cgmartini.nl), extract it, and copy the `martini_v300` folder into your `INIT_CG` directory.

#### 5. Run the Simulation
Finally, execute the simulation using the provided Python script:

```bash
python run.py
```
