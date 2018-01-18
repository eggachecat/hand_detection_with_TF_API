# 训练手的分类(左/右)
## 生资料
### 使用
> python transform_data.py {data_path} [--path_type {path_type}] [--fixed_aspect_ratio {fixed_aspect_ratio}]
### 说明
生成的资料会在当前的data文件夹下分为train和test
### 参数
`{data_path}` 是HTC的資料夾的位置(該位置應該有三個文件夾:DeepQ-Synth-Hand-01, DeepQ-Synth-Hand-02, DeepQ-Viewpaper)

`{path_type}` 是用到多少资料,可以看code,默认0用所有的资料(synth+viewpaper)

`fixed_aspect_ratio` 是从正确答案中取出手resize的时候要不要保持比例,默认是True

## 训练
## 使用
> python train.py
