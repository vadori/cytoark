# CYTOARK: Cytoarchitecture analysis in Python
<div>
  <img width="50%" src="https://github.com/Vadori/cytoark/assets/36676465/8bed4528-ffea-49c8-81b9-e457b0d32bf3" alt="Sample image compressed 50%" title="Sample image compressed 50%">
<div>

**Cytoark** aims to collect tools for the automatic analysis of histological data. The focus is on the cytoarchitecture of the brain for comparative neuroanatomy. The main component right now is **CISCA**, an instance segmentation and classification model described 
[here](https://www.arxiv.org/abs/2409.04175).

## CISCA

Delineating and classifying individual cells in microscopy tissue images is a complex task, yet it is a pivotal endeavor in various medical and biological investigations. We propose a new deep learning framework (CISCA) for automatic cell instance segmentation and classification in histological slices to support detailed morphological and structural analysis or straightforward cell counting in digital pathology workflows and brain cytoarchitecture studies. At the core of CISCA lies a network architecture featuring a lightweight U-Net with three heads in the decoder. The first head classifies pixels into boundaries between neighboring cells, cell bodies, and background, while the second head regresses four distance maps along four directions. The network outputs from the first and second heads are integrated through a tailored post-processing step, which ultimately yields the segmentation of individual cells. A third head enables simultaneous classification of cells into relevant classes, if required. We showcase the effectiveness of our method using four datasets, including CoNIC, PanNuke, and MoNuSeg, which are publicly available H\&E datasets. Additionally, we introduce CytoDArk0, a novel dataset consisting of Nissl-stained images of the cortex, cerebellum, and hippocampus from mammals belonging to the orders Cetartiodactyla and Primates. We evaluate CISCA in comparison to other state-of-the-art methods, demonstrating CISCA's robustness and accuracy in segmenting and classifying cells across diverse tissue types, magnifications, and staining techniques.

![image](https://github.com/user-attachments/assets/c38a5ed4-080d-42bd-8947-d1788641203e)
![process_diagram](https://github.com/user-attachments/assets/e90d6022-1b58-4c6f-9358-802cf17b74e2)
![network_diagram](https://github.com/user-attachments/assets/1a8673b4-fb60-46f1-8398-514299a1ca65)

## Getting Started

### Dependencies

### Installing

### Executing program

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments



