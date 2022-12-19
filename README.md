# SemiSupGrassClover
Code for the ICCVW CVPPA 2021 paper: Semi-supervised dry herbage mass estimation using automatic data and synthetic images [paper](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Albert_Semi-Supervised_Dry_Herbage_Mass_Estimation_Using_Automatic_Data_and_Synthetic_ICCVW_2021_paper.pdf)


# How to train
## Generate the synthetic images
The first step is to generate the synthetic data using synthetic_gen/generate_syn.py 
To generate the synthetic data, you will need a set of background images and individually cropped elements organized in cropedbit/grass cropedbit/clover
cropedbit/weeds
Use python generate_syn.py 8000 to generate 8000 synthetic images and the associated ground-truth

## Train the segmentation model on the synthetic data
Code is present in the semseg/ folder to train the sementic segmentation model. See the .sh files and set the path to the synthetic images in the semseg/mypath.py file.
There is code in the semseg/train.sh file to use the trained semseg model to predict pixel percentages in the canopy and to train a linear classifier to predict the dry biomass and store it in a biomasscomposition.csv file. This file is to be used to train on the unlabeled images.

## Train on the labeled data + estimated ground-truth for the unlabeled images
The train.sh file in the root directory lists comands to train the final regression CNN using or not the automatically labeled images.

Feel free to contact me/open an issue for further informations

# Cite our work
If our work helped your research:

```
@inproceedings{2021_ICCVW_semisupclover,
  title={Semi-supervised dry herbage mass estimation using automatic data and synthetic images},
  author={Albert, Paul and Saadeldin, Mohamed and Narayanan, Badri and Mac Namee, Brian and Hennessy, Deirdre and O'Connor, Aisling and O'Connor, Noel and McGuinness, Kevin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1284--1293},
  year={2021}
}
```

Albert, Paul, et al. "Semi-supervised dry herbage mass estimation using automatic data and synthetic images." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
