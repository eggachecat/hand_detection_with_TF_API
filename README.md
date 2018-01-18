# 訓練
分別參看`hand_detection` 和 `hand_classification` 的README

已經train好的模型在models目錄下

# 測試
## local
### 指令
> python inference.py --local 1 [--infer_data {{infer_data_path}}]

若沒有--infer_data則將test_images中的圖片輸出到outputs(label是分類器給的)

若有,例如將frame的結果輸出,則是例如
`--infer_data ./DeepQ-Vivepaper/frame/air ./DeepQ-Vivepaper/frame/book`

## HTC submit
### 打包
bash pack.sh

### 上傳
填寫 Executing commands: bash run.sh

### 註釋
由於網絡速度的原因,在htc上測試的模型是從dropbox下載(見download.sh的地址)