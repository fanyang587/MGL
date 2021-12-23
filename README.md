# Mutual Graph Learning for Camouflaged Object Detection (CVPR2021)

> Authors:
> [Qiang Zhai](https://github.com/cvqiang/mgl), 
> [Xin Li](https://scholar.google.com/citations?user=TK-hRO8AAAAJ&hl=en), 
> [Fan Yang](https://scholar.google.com/citations?user=FSfSgwQAAAAJ&hl=en), 
> [Chenglizhao Chen](https://scholar.google.com/citations?user=SGjgjBUAAAAJ&hl=zh-CN), 
> [Hong Cheng](https://scholar.google.com/citations?user=-845MAcAAAAJ&hl=zh-CN), 
> [Deng-Ping Fan](https://dpfan.net/).



1. Configuring your environment (Prerequisites):
    The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch).   
    
    Note that MGLNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda env create -f env.yaml`.
    
    
<!--2. Downloading Testing Sets: -->
2. Downloading Testing Sets:
    + downloading _**NEW testing dataset**_ (COD10K-test + CAMO-test + CHAMELEON), which can be found in this [Google Drive link](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/143yHFLAabMBT7wgXA0LrMg) with the fetch code: z83z.
    <!--
    + download **_NEW training dataset_** (COD10K-train) which can be found in this [Google Drive link](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1XL6OjpDF-MVnXOY6-bdaBg) with the fetch code:djq2.  Please refer to our original paper for other training data. -->
    

<!--3. Training Configuration:

    + Assigning your customed path, like `--save_model`, `--train_img_dir`, and `--train_gt_dir` in `MyTrain.py`.
    
    + Just run it! -->

3. Testing Configuration:

    + After you download all the trained models [Google Drive link](https://drive.google.com/file/d/1KCYYcb3UM8a9Hg71f2KbowpdxGhzz6tu/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1pfFyOhOiaJVIxGjg6TFPLg) with the fetch code: ry8h, move it into './model_file/', and testing data.
    + Assigning your comstomed path in 'config/cod_mgl50.yaml', like 'data_root', 'test_list'.
    + Ensure consistency between 'stage' and 'model_path'. Setting 'stage: 1' and 'model_path: pre-trained/mgl_s.pth' to evaluate S-MGL model and setting 'stage: 2' and 'model_path: pre-trained/mgl_r.pth' to evaluate R-MGL model.
    + Playing 'test.py' to generate the final prediction map, the predicted camouflaged object region and cmouflaged object edge is saved into 'exp/result' as default.
    + You can also download the results [Google Drive link](https://drive.google.com/file/d/1Gi8JVgl3MFj3GCIW9FeE1gcmgCCzMI77/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1-0RV5ZORNznN_OVFTOoyNw) with the fetch code: b1gr.
4. Other Dataset:
    + For NC4K dataset: You can find the results in [Google Drive link](https://drive.google.com/file/d/1EgfD_GtxTlP7CSJI9RRQuKhjhbsg2DZy/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1Czgs3RciBZQjw0CB9iPPWw) with the fetch code: 8ntb.
    
5. Evaluation your trained model:

    + One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
    please follow this the instructions in `main.m` and just run it to generate the evaluation results in 
    `./EvaluationTool/EvaluationResults/Result-CamObjDet/`.

6. Training Configuration:
   + After you download the initial model [Google Drive link](https://drive.google.com/file/d/17WYyKg40DkAgFWOusiAKgqZOlfUFzjn5/view?usp=sharing) or Baidu Pan link, move it to './pre_trained/'.
   + Put the 'train_test_file/train.lst' to the path which is included in cod_mgl50.yaml.
   + Run train.py

7. If you think this work is helpful, please cite

```
@inproceedings{zhai2021Mutual,
  title={Mutual Graph Learning for Camouflaged Object Detection},
  author={Zhai, Qiang and Li, Xin and Yang, Fan and Chen, Chenglizhao and Cheng, Hong and Fan, Deng-Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2021}
}
```

