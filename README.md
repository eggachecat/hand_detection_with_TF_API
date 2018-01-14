# 本地测试
## 指令
> python inference.py --local 1 [--infer_data {{infer_data_path}}]

若没有--infer_data则将test_images中的图片输出到outputs(label是分类器给的)

若有,例如将frame的结果输出,则是例如
`--infer_data ./DeepQ-Vivepaper/frame/air ./DeepQ-Vivepaper/frame/book`

# HTC submit
## 打包
bash pack.sh
## 上传
Executing commands: bash run.sh