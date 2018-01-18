
# 訓練手的分類(左/右)
## 生資料
### 使用
> python transform_data.py {data_path} [--path_type {path_type}] [--fixed_aspect_ratio {fixed_aspect_ratio}]
### 說明
生成的資料會在當前的data文件夾下分為train和test
### 參數
`{data_path}` 是HTC的資料夾的位置(該位置應該有三個文件夾:DeepQ-Synth-Hand-01, DeepQ-Synth-Hand-02, DeepQ-Viewpaper)

`{path_type}` 是用到多少資料,可以看code,默認0用所有的資料(synth+viewpaper)

`fixed_aspect_ratio` 是從正確答案中取出手resize的時候要不要保持比例,默認是True

## 訓練
### 使用
> python train.py
