# Zero-Shot Point Cloud Segmentation by Semantic-Visual Aware Synthesis
This is the code repository related to "[Zero-Shot Point Cloud Segmentation by Semantic-Visual Aware Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Zero-Shot_Point_Cloud_Segmentation_by_Semantic-Visual_Aware_Synthesis_ICCV_2023_paper.html)" (ICCV 2023, Poster) in PyTorch implementation. 

## 1. Abstract
This paper proposes a feature synthesis approach for zero-shot semantic segmentation of 3D point clouds, enabling generalization to previously unseen categories. Given only the class-level semantic information for unseen objects, we strive to enhance the correspondence, alignment and consistency between the visual and semantic spaces, to synthesise diverse, generic and transferable visual features. We develop a masked learning strategy to promote diversity within the same class visual features and enhance the separation between different classes. We further cast the visual features into a prototypical space to model their distribution for alignment with the corresponding semantic space. Finally, we develop a consistency regularizer to preserve the semantic-visual relationships between the real seen features and synthetic-unseen features. Our approach shows considerable semantic segmentation gains on ScanNet, S3DIS and SemanticKITTI benchmarks.

# To-do List
Release our training and test code on 3 datasets before Decemeber 10th, 2023 (As soon as we can, after CVPR 2024);

Release our pretrained models within December;

## Citation
If it is helpful to your research, please cite our paper as follows:

    @inproceedings{yang2023zero,
         title={Zero-Shot Point Cloud Segmentation by Semantic-Visual Aware Synthesis},
         author={Yang, Yuwei and Hayat, Munawar and Jin, Zhao and Zhu, Hongyuan and Lei, Yinjie},
         booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
         pages={11586--11596},
         year={2023}
     }



