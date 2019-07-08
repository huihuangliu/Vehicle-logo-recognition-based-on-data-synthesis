### 文件目录
#### 获取阈值（可选步骤）
在ensemble.py中我们用到了阈值，这个阈值是通过calculate\_rcnn2\_mAP.py以及calculate\_rcnn3\_mAP.py生成的。
1. 将模型py-faster-rcnn2\_15000\_iters100000的验证结果放入./results/py-faster-rcnn2_15000_iters100000test53107\_12\_10/Main/下
2. 将模型py-faster-rcnn3\_15000\_iters100000的验证结果放入./results/py-faster-rcnn3_15000_iters100000test53107\_12\_10/Main/下
3. 将验证集的标签放入./code/data/validate_annotations/Annotations/下
4. 分别运行calculate\_rcnn2\_mAP.py和calculate\_rcnn3\_mAP.py生成对应的模型的阈值写入ensemble.py中


#### 模型融合
1. 将模型py-faster-rcnn2\_15000\_iters100000的测试结果放入./results/py-faster-rcnn2_15000_iters100000test53107\_12\_10/Main/下
2. 将模型py-faster-rcnn3\_15000\_iters100000的测试结果放入./results/py-faster-rcnn3_15000_iters100000test53107\_12\_10/Main/下
3. 拼接的结果放到./results/results12\_7\_pycnn\_5\_7\_100000/下
4. 修改ensemble.py中的TEST_ROOT为测试集路径
5. 运行ensemble.py，生成结果文件路径为./results/result.json（PS: 生成的json顺序可能和已提交最高结果不一样，但里面每张图片的标签是一样的）
