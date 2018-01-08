# duckduckduck
## data_utils
### 說明
準備資料給Tensorflow Object Detection API使用
### 使用
> python data_utils.py {data_path}
### 參數
`{data_path}` 是HTC的資料夾的位置(該位置應該有三個文件夾:DeepQ-Synth-Hand-01, DeepQ-Synth-Hand-02, DeepQ-Viewpaper)
### 結果
在`./training`文件下的`train`和`eval`兩個文件夾下應該各有0~11個tfrecord文件,其中train是會被API那去訓練,eval是拿去做validation.默認訓練資料比例是80%,可以調`TRAIN_RATIO`來改變.

會產生`data_configs.json`在當前文件夾,可以檢查下資料是否有錯

## Tensorflow Object Detection API
### 安裝
參考這個[youtube](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku)
### 資料
將`training`這個文件夾移到`{你的到models目錄}/research/objection_detection/`這個目錄下

訓練和網絡的配置在`traing/train.config`這裡.包括batch_size等參數可以調整
### 開始訓練
>cd {你的到models目錄}/research/objection_detection/

>python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/train.config
### 測試
1. 將訓練的模型保存到tensorflow能讀取的樣子
> python export_inference_graph.py     --input_type image_tensor    --pipeline_config_path training/train.config  --trained_checkpoint_prefix training/model.ckpt-{你選的保存的值} --output_directory hand_detection

2. 將`object_detection_test.ipynb`移到`{你的到models目錄}/research/objection_detection/`

3. 將需要測試的圖片放到`{你的到models目錄}/research/objection_detection/test_images/`下

4. 打開object_detection_test.ipynb
>jupyter notebook 
