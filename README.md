# CVIU-2S-SGCN
[CVIU-2024] - 3D Face Landmark Detection with Two Stage-Stratified Graph Convolutional Neural Network

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Facial Expression Analysis](https://img.shields.io/badge/dataset-FACESCAPE-green)](https://facescape.nju.edu.cn) [![Facial Expression Analysis](https://img.shields.io/badge/dataset-HEADSPACE-green)](https://www-users.york.ac.uk/~np7/research/Headspace/)
[![pytorch-geometric](https://img.shields.io/badge/Framework-PyTorch_Geometric-red)](https://pytorch-geometric.readthedocs.io/en/latest/)

## Description

Facial Landmark Detection (FLD) algorithms play a pivotal role in numerous computer vision applications, particularly in tasks like face recognition, head pose estimation, and facial expression analysis. While 2D FLD has long been the focus, the emergence of 3D data has led to a surge of interest in 3D FLD due to its potential applications in various fields, including medical research. However, automating 3D FLD presents significant challenges, including selecting suitable network architectures, refining outputs for precision, and optimizing computational efficiency. In response, this paper presents a novel approach, the 2-Stage Stratified Graph Convolutional Network (2S-SGCN), which addresses these challenges comprehensively. The proposed method incorporates a heatmap regression stage leveraging both local and long-range dependencies through a stratified approach. This design not only enhances performance but also ensures suitability for resource-constrained devices. In the second stage, 3D landmarks are accurately determined using a novel post-processing technique based on geodesic distances on the original mesh. Experimental results on Facescape and Headspace public datasets demonstrate that the proposed method achieves state-of-the-art performance under various conditions, showcasing its efficacy in real-world scenarios.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install and set up the project, follow these steps:

1. Create a virtual environment using `venv`:

   ```bash
   python -m venv myenv
   ```

2. Activate the virtual environment:

   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source myenv/bin/activate
     ```

3. Install the required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, follow these steps:

1. Activate the virtual environment (if not already activated):
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source myenv/bin/activate
     ```

### Train/Test on FACESCAPE Dataset

After requesting FACESCAPE Dataset [here](https://facescape.nju.edu.cn), you can replicate exact experiments as described in the paper.
you need to recreate the following .data dir structure

```
.
.
└── .data/
│   └── landmark_indices.npz
│   └── processed/
│   └── raw/
│   │   └── 1/
│   │   |   └── 1_neutral.jpg
│   │   |   └── 1_neutral.obj
│   │   |   └── 1_neutral.obj.mtl
│   │   |   └── 2_smile.jpg
│   │   |   └── 2_smile.obj
│   │   |   └── 2_smile.obj.mtl
│   │   |   └── 4_anger.jpg
│   │   |   └── 4_anger.obj
│   │   |   └── 4_anger.obj.mtl
│   │   |   └── 14_sadness.jpg
│   │   |   └── 14_sadness.obj
│   │   |   └── 14_sadness.obj.mtl
│   │   └── 2/
│   │   |   └── 1_neutral.jpg
│   │   |   └── ...
│   │   |   └── 14_sadness.obj.mtl
│   │   └── .../
│   │   └── 847/
...
```

Then you can execute the train/test with the parameters in the `config.py` file with the following command:

```bash
python main.py --do_refine=True
```
where --do_refine=True is for the refinement procedure to get landmarks on the 3D surface

Otherwise if you want to perform the refinement procedure separately you can execute:

```bash
python refine.py --path=<path-to-model>
```

## Contributing

To contribute to the project, please follow these guidelines:

1. Fork the repository and clone it to your local machine.

2. Create a new branch for your feature or bug fix.

3. Make your changes and commit them with descriptive commit messages.

4. Push your branch to your forked repository.

5. Submit a pull request to the main repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact
