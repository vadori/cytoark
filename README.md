
# CYTOARK: Cytoarchitecture analysis in Python (WIP!)

<img width="20%" src="https://github.com/Vadori/cytoark/assets/36676465/8bed4528-ffea-49c8-81b9-e457b0d32bf3" alt="cytoark" title="cytoark" align="right">

**Cytoark** aims to provide tools for the automatic analysis of histological data. The focus is on analyzing the brain cytoarchitecture for comparative neuroanatomy studies. The main components right now are the following:
- **CISCA**, a cell instance segmentation and classification model. The codebase is available in this repo. 
- **CytoDArk0**, the first Nissl-stained histological dataset of the mammalian brain with annotations of single cells.  CytoDArk0 is available on Zenodo [CytoDArk0](https://zenodo.org/records/13694738).

Both are described [here](https://www.arxiv.org/abs/2409.04175) (pre-print).

WIP:
- [**Lace**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10635252), the first self-supervised attempt at detecting layers in 2D Nissl-stained histological slices of the cerebral cortex 
- [**C-Shape**](https://arxiv.org/abs/2411.00561), a comprehensive comparison of methods for classifiyng cell shapes 
  
Please note that this repo is WIP, so please contact the owner in case you run into any issues. 

## CISCA: a Cell Instance Segmentation and Classification method for histo(patho)logical image Analyses

Delineating and classifying individual cells in microscopy tissue images is a complex task, yet it is a pivotal endeavor in various medical and biological investigations. We propose a new deep learning framework (CISCA) for automatic cell instance segmentation and classification in histological slices to support detailed morphological and structural analysis or straightforward cell counting in digital pathology workflows and brain cytoarchitecture studies. At the core of CISCA lies a network architecture featuring a lightweight U-Net with three heads in the decoder. The first head classifies pixels into boundaries between neighboring cells, cell bodies, and background, while the second head regresses four distance maps along four directions. The network outputs from the first and second heads are integrated through a tailored post-processing step, which ultimately yields the segmentation of individual cells. A third head enables simultaneous classification of cells into relevant classes, if required. We showcase the effectiveness of our method using four datasets, including CoNIC, PanNuke, and MoNuSeg, which are publicly available H\&E datasets. Additionally, we introduce CytoDArk0, a novel dataset consisting of Nissl-stained images of the cortex, cerebellum, and hippocampus from mammals belonging to the orders Cetartiodactyla and Primates. We evaluate CISCA in comparison to other state-of-the-art methods, demonstrating CISCA's robustness and accuracy in segmenting and classifying cells across diverse tissue types, magnifications, and staining techniques.

<img width="37%" src="https://github.com/user-attachments/assets/c38a5ed4-080d-42bd-8947-d1788641203e" alt="results" title="results" align="left">
<img width="58%" src="https://github.com/user-attachments/assets/e90d6022-1b58-4c6f-9358-802cf17b74e2" alt="diagram" title="diagram" align="right">

![network_diagram](https://github.com/user-attachments/assets/1a8673b4-fb60-46f1-8398-514299a1ca65)

[^Comment]:  ## Getting Started

[^Comment]:  ### Dependencies

[^Comment]:  ### Installing

[^Comment]:  ### Executing program

## License

This project is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) License - see the LICENSE.md file for details.

## Acknowledgments
We thank all the authors who inspired us and shared their work. In particular, we acknowledge the following:
- Schmidt, Uwe, et al. "Cell detection with star-convex polygons." Medical Image Computing and Computer Assisted Intervention–MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11. Springer International Publishing, 2018. [Code](https://github.com/stardist/stardist) [Paper](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_30)
- Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical image analysis 58 (2019): 101563. [Tensorflow original code](https://github.com/vqdang/hover_net/tree/tensorflow-final) [Paper](https://www.sciencedirect.com/science/article/pii/S1361841519301045)
- Upschulte, Eric, et al. "Contour proposal networks for biomedical instance segmentation." Medical image analysis 77 (2022): 102371. [Code](https://github.com/FZJ-INM1-BDA/celldetection/tree/main) [Paper](https://www.sciencedirect.com/science/article/pii/S136184152200024X)
- Hörst, Fabian, et al. "Cellvit: Vision transformers for precise cell segmentation and classification." Medical Image Analysis 94 (2024): 103143. [Code](https://github.com/TIO-IKIM/CellViT) [Paper](https://www.sciencedirect.com/science/article/pii/S1361841524000689)
- CSBDeep - a deep learning toolbox for microscopy image restoration and analysis [Code](https://github.com/CSBDeep/CSBDeep)

## Attribution
If you find our dataset or CISCA framework useful, we'd love a shoutout! Here’s a citation format you can use:

**DATASET (cytoDArk0) and/or CELL SEGMENTATION & CLASSIFICATION METHOD (CISCA)**

- Vadori, V., Graïc, J.-M., Peruffo, A., Vadori, G., Finos, L., & Grisan, E. (2024). CISCA and CytoDArk0: a Cell Instance Segmentation and Classification method for histo(patho)logical image Analyses and a new, open, Nissl-stained dataset for brain cytoarchitecture studies. arXiv preprint arXiv:2409.04175. https://doi.org/10.48550/arXiv.2409.04175
- @article{vadori2024cisca,
  author = {Vadori, Valentina and
            Graïc, Jean-Marie and
            Peruffo, Antonella and
            Vadori, Giulia and
            Finos, Livio and
            Grisan, Enrico},
  title = {CISCA and CytoDArk0: a Cell Instance Segmentation and Classification method for histo(patho)logical image Analyses and a new, open, Nissl-stained dataset for brain cytoarchitecture studies},
  year = {2024},
  journal={arXiv e-prints},
  pages={arXiv--2409},
}

**DATASET (cytoDArk0)**

- Vadori, V., Graïc, J.-M., Peruffo, A., Vadori, G., Finos, L., & Grisan, E. (2024). CytoDark0 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13694738
- @dataset{vadori2024cytodark0,
  author = {Vadori, Valentina and
            Graïc, Jean-Marie and
            Peruffo, Antonella and
            Vadori, Giulia and
            Finos, Livio and
            Grisan, Enrico},
  title = {CytoDArk0},
  month = sep,
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.13694738},
  url = {https://doi.org/10.5281/zenodo.13694738}
}






