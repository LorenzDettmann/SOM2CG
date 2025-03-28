# Granulate - Coarse-graining soil organic matter models

A tool for converting atomistic soil organic matter (SOM) models from the Vienna soil organic matter modeler 2 (VSOMM2) to a coarse-grained representation, compatible with the Martini 3 force field.

## Prerequisites

The script is written in Python 3. The following packages are needed and can be installed via `pip`:
```bash
pip install mdanalysis rdkit pyyaml
```

## Execution

You can execute the script in the directory with the topology files from the VSOMM2, or modify e.g. the directory paths using the following arguments:
```bash
python3 granulate.py [-V] [-h] [-nt 'number of threads'] 
                     [-input_dir 'input directory'] 
                     [-output_dir 'output directory'] 
                     [-n_confs 'number of conformers'] 
                     [-map '"cog" or "com" mapping'] 
                     [-parametrize '(yes/no) parametrize the molecules, or only output mapped structure file']
                     [-use_std_fc '(yes/no) use standard force constants']
                     [-with_progress_bar '(yes/no) activate a progress bar']
                     [-config 'YAML configuration file']
```

## Contributing
If you have any suggestions for improving the efficiency of the script, or for additional features, feel free to contact one of the authors of the publication.

## References and Acknowledgments
If you use Granulate, please cite the following papers:
```
Dettmann, L. F.; Kühn, O.; Ahmed, A. A. Automated Parametrization Approach for Coarse-Graining Soil Organic Matter Molecules. Journal of Chemical Theory and Computation, 2024, 20, 10684–10696. https://doi.org/10.1021/acs.jctc.4c01334. 

Dettmann, L. F.; Kühn, O.; Ahmed, A. A. Martini-Based Coarse-Grained Soil Organic Matter Model Derived from Atomistic Simulations. Journal of Chemical Theory and Computation, 2024, 20, 5291–5305. https://doi.org/10.1021/acs.jctc.4c00332. 
```

Parts of this code are based on work by Mark A. Miller and coworkers, used with permission.
These parts are subject to the following citations:
```
Potter, T. D.; Haywood, N.; Teixeira, A.; Hodges, G.; Barrett, E. L.; Miller, M. A. Partitioning into Phosphatidylcholine–Cholesterol Membranes: Liposome Measurements, Coarse-Grained Simulations, and Implications for Bioaccumulation. Environmental Science: Processes &amp; Impacts, 2023, 25, 1082–1093. https://doi.org/10.1039/d3em00081h. 
    
Potter, T. D.; Barrett, E. L.; Miller, M. A. Automated Coarse-Grained Mapping Algorithm for the Martini Force Field and Benchmarks for Membrane–Water Partitioning. Journal of Chemical Theory and Computation, 2021, 17, 5777–5791. https://doi.org/10.1021/acs.jctc.1c00322. 
```
Please cite these works if you use this code in your research.
We thank Mark. A. Miller and coworkers for their contributions.

## License
This software is distributed under the MIT license.

    Copyright (c) 2024 Lorenz Friedrich Dettmann
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
