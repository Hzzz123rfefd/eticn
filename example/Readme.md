# 文章涉及到的复现脚本
## 一、训练模型
### **1.1 eticn系列模型训练**

- **eticn基座模型**：python train.py --model_config_path config/eticn/eticn.yml
- **eticn:**

```bash
python train.py --model_config_path config/eticn/eticn_013.yml
python train.py --model_config_path config/eticn/eticn_0018.yml
python train.py --model_config_path config/eticn/eticn_025.yml
python train.py --model_config_path config/eticn/eticn_0067.yml
python train.py --model_config_path config/eticn/eticn_0483.yml
python train.py --model_config_path config/eticn/eticn_0932.yml
```
- **eticn-qeevrf:** ``` python train.py --model_config_path config/eticn/eticn_qeevrf_stage2.yml & python train.py --model_config_path config/eticn/eticn_qeevrf_stage3.yml ```
- **eticn-qvrf:** ``` python train.py --model_config_path config/eticn/eticn_qvrf.yml ```
- **eticn-vgvrf:** ``` python train.py --model_config_path config/eticn/eticn_vgvrf.yml ```
- **eticn-stvrf:** ``` python train.py --model_config_path config/eticn/eticn_stvrf.yml ```
- **eticn-msd:** ``` python train.py --model_config_path config/eticn/eticn_msd.yml ```

### 1.2 **gric系列模型训练**

- **gric基座模型:** `python train.py --model_config_path config/gric/gric.yml`
- **gric**

```bash
python train.py --model_config_path config/gric/gric_013.yml
python train.py --model_config_path config/gric/gric_0018.yml
python train.py --model_config_path config/gric/gric_025.yml
python train.py --model_config_path config/gric/gric_0067.yml
python train.py --model_config_path config/gric/gric_0483.yml
python train.py --model_config_path config/gric/gric_0932.yml
```

- **gric-qeevrf:** ``` python train.py --model_config_path config/gric/gric_qeevrf_stage2.yml & python train.py --model_config_path config/gric/gric_qeevrf_stage3.yml ```
- **gric-qvrf:** ``` python train.py --model_config_path config/gric/gric_qvrf.yml ```
- **gric-vgvrf:** ``` python train.py --model_config_path config/gric/gric_vgvrf.yml ```
- **gric-stvrf:** ``` python train.py --model_config_path config/gric/gric_stvrf.yml ```

### **1.3 stf系列模型训练**

- **stf基座模型:** `python train.py --model_config_path config/stf/stf.yml`
- **stf**

```bash
python train.py --model_config_path config/stf/stf_013.yml
python train.py --model_config_path config/stf/stf_0018.yml
python train.py --model_config_path config/stf/stf_025.yml
python train.py --model_config_path config/stf/stf_0067.yml
python train.py --model_config_path config/stf/stf_0483.yml
python train.py --model_config_path config/stf/stf_0932.yml
```

- **stf-qeevrf**:``` python train.py --model_config_path config/stf/stf_qeevrf_stage2.yml & python train.py --model_config_path config/stf/stf_qeevrf_stage3.yml ```
- **stf-qvrf**:``` python train.py --model_config_path config/stf/stf_qvrf.yml ```
- **stf-vgvrf**: ``` python train.py --model_config_path config/stf/stf_vgvrf.yml ```
- **stf-stvrf:** ``` python train.py --model_config_path config/stf/stf_stvrf.yml ```
- **stf-msd:** ``` python train.py --model_config_path config/stf/stf_msd.yml ```

### **1.4 vaic系列模型训练**

- **vaic基座模型:** `python train.py --model_config_path config/vaic/vaic.yml`
- **vaic**

```bash
python train.py --model_config_path config/vaic/vaic_013.yml
python train.py --model_config_path config/vaic/vaic_0018.yml
python train.py --model_config_path config/vaic/vaic_025.yml
python train.py --model_config_path config/vaic/vaic_0067.yml
python train.py --model_config_path config/vaic/vaic_0483.yml
python train.py --model_config_path config/vaic/vaic_0932.yml
```

- vaic-qeevrf: ``` python train.py --model_config_path config/vaic/vaic_qeevrf_stage2.yml & python train.py --model_config_path config/vaic/vaic_qeevrf_stage3.yml ```
- vaic-qvrf:``` python train.py --model_config_path config/vaic/vaic_qvrf.yml ```
- vaic-stvrf: ``` python train.py --model_config_path config/vaic/vaic_stvrf.yml ```
- vaic-vgvrf: ``` python train.py --model_config_path config/vaic/vaic_vgvrf.yml ```
- vaic-msd: ``` python train.py --model_config_path config/vaic/vaic_msd.yml ```

### **1.5 elic系列模型训练**

- **elic基座模型:** `python train.py --model_config_path config/elic/elic.yml`
- **elic**

```bash
python train.py --model_config_path config/elic/elic_013.yml
python train.py --model_config_path config/elic/elic_0018.yml
python train.py --model_config_path config/elic/elic_025.yml
python train.py --model_config_path config/elic/elic_0067.yml
python train.py --model_config_path config/elic/elic_0483.yml
python train.py --model_config_path config/elic/elic_0932.yml
```



## 二、评估模型
### 2.1 固定速率
#### 2.1.1 camvid数据集
* eticn: ```python example/eval.py --model_config_path config/eticn/eticn.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/ --save_path result/R-D/fbr/camvid/[Ours].json```
* elic: ```python example/eval.py --model_config_path config/elic/elic.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/elic/ --save_path result/1.json```
* stf：```python example/eval.py --model_config_path config/stf/stf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/ --save_path result/R-D/fbr/camvid/[Zou(CVPR2022)].json ```
* gric:```python example/eval.py --model_config_path config/gric/gric.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/gric/ --save_path result/R-D/fbr/camvid/[Yang(NIPS2024)].json ```
* vaic: ```python example/eval.py --model_config_path config/vaic/vaic.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vaic/ --save_path result/R-D/fbr/camvid/[Minnen(NIPS2018)].json```
* jepg2000：```python example/eval_traditional.py --type jepg --data_path datasets/camvid/camvid_train/val.jsonl  --result_path result/R-D/fbr/camvid/[JEPG2000].json```
* bpg：```python example/eval_traditional.py --type bpg --data_path datasets/camvid/camvid_train/val.jsonl  --result_path result/R-D/fbr/camvid/[BPG].json```
* vvc：```python example/eval_traditional.py --type vvc --data_path datasets/camvid/camvid_train/val.jsonl  --result_path result/R-D/fbr/camvid/[VVC].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/fbr/camvid --base_result_file [BPG].json```
* RD曲线：```python example/plot_R_D_curve.py --dir_path result/R-D/fbr/camvid```

#### 2.1.2 soda数据集
* **eticn**：```python example/eval.py --model_config_path config/eticn/eticn.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/eticn/ --save_path result/R-D/fbr/soda/[Ours].json```
* **stf**：```python example/eval.py --model_config_path config/stf/stf.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/stf/ --save_path result/R-D/fbr/soda/[Zou(CVPR2022)].json ```
* **gric**:```python example/eval.py --model_config_path config/gric/gric.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/gric/ --save_path result/R-D/fbr/soda/[Yang(NIPS2024)].json ```
* **vaic**: ```python example/eval.py --model_config_path config/vaic/vaic.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/vaic/ --save_path result/R-D/fbr/soda/[Minnen(NIPS2018)].json```
* **jepg2000**：```python example/eval_traditional.py --type jepg --data_path datasets/soda/soda_train/val.jsonl  --result_path result/R-D/fbr/soda/[JEPG2000].json```
* **bpg**：```python example/eval_traditional.py --type bpg --data_path datasets/soda/soda_train/val.jsonl  --result_path result/R-D/fbr/soda/[BPG].json```
* **vvc**：```python example/eval_traditional.py --type vvc --data_path datasets/soda/soda_train/val.jsonl  --result_path result/R-D/fbr/soda/[VVC].json```
* **RD曲线**：```python example/plot_R_D_curve.py --dir_path result/R-D/fbr/soda```
* **BD结果**：```python example/get_bp_psnr_rate.py --dir_path result/R-D/fbr/soda --base_result_file [BPG].json```

#### 2.1.3 srti数据集
* **eticn**：```python example/eval.py --model_config_path config/eticn/eticn.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/eticn/ --save_path result/R-D/fbr/srti/[Ours].json```
* **stf**：```python example/eval.py --model_config_path config/stf/stf.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/stf/ --save_path result/R-D/fbr/srti/[Zou(CVPR2022)].json ```
* **gric**:```python example/eval.py --model_config_path config/gric/gric.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/gric/ --save_path result/R-D/fbr/srti/[Yang(NIPS2024)].json```
* **vaic**: ```python example/eval.py --model_config_path config/vaic/vaic.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/vaic/ --save_path result/R-D/fbr/srti/[Minnen(NIPS2018)].json```
* **jepg2000**：```python example/eval_traditional.py --type jepg --data_path datasets/srti/srti_train/val.jsonl  --result_path result/R-D/fbr/srti/[JEPG2000].json```
* **bpg**：```python example/eval_traditional.py --type bpg --data_path datasets/srti/srti_train/val.jsonl  --result_path result/R-D/fbr/srti/[BPG].json```
* **vvc**：```python example/eval_traditional.py --type vvc --data_path datasets/srti/srti_train/val.jsonl  --result_path result/R-D/fbr/srti/[VVC].json```
* **BD结果**：```python example/get_bp_psnr_rate.py --dir_path result/R-D/fbr/srti --base_result_file [BPG].json```
* **RD曲线**：```python example/plot_R_D_curve.py --dir_path result/R-D/fbr/srti```

bdd数据集：
* eticn：```python example/eval.py --model_config_path config/eticn/eticn.yml --data_path datasets/bdd/bdd_train/val.jsonl --model_path saved_model/eticn/ --save_path result/R-D/fbr/bdd/[Ours].json```
* stf：```python example/eval.py --model_config_path config/stf/stf.yml --data_path datasets/bdd/bdd_train/val.jsonl --model_path saved_model/stf/ --save_path result/R-D/fbr/bdd/[Zou(CVPR2022)].json ```
* gric:```python example/eval.py --model_config_path config/gric/gric.yml --data_path datasets/bdd/bdd_train/val.jsonl --model_path saved_model/gric/ --save_path result/R-D/fbr/bdd/[Yang(NIPS2024)].json ```
* vaic: ```python example/eval.py --model_config_path config/vaic/vaic.yml --data_path datasets/bdd/bdd_train/val.jsonl --model_path saved_model/vaic/ --save_path result/R-D/fbr/bdd/[Minnen(NIPS2018)].json```
* jepg2000：```python example/eval_traditional.py --type jepg --data_path datasets/bdd/bdd_train/val.jsonl  --result_path result/R-D/fbr/bdd/[JEPG2000].json```
* bpg：```python example/eval_traditional.py --type bpg --data_path datasets/bdd/bdd_train/val.jsonl  --result_path result/R-D/fbr/bdd/[BPG].json```
* vvc：```python example/eval_traditional.py --type vvc --data_path datasets/bdd/bdd_train/val.jsonl  --result_path result/R-D/fbr/bdd/[VVC].json```

### 2.2 可变速率

#### 2.2.1 Camvid数据集

##### 2.2.1 eticn为基座模型
* **qeevrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qeevrf_stage3.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_qeevrf/stage3 --save_path result/R-D/vbr/eticn/[Ours].json```
* **qvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_qvrf/stage2 --save_path result/R-D/vbr/eticn/[Tong(NIPS2023)].json```
* **vgvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_vgvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_vgvrf/ --save_path result/R-D/vbr/eticn/[Cai(CVPR2022)].json```
* **stvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_stvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_stvrf/ --save_path result/R-D/vbr/eticn/[Presta(TIP2025)].json```
* **msd**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_msd.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_msd/ --save_path result/R-D/vbr/eticn/camvid/[MSD].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/eticn --base_result_file Baseline[ETICN].json```
* RD结果：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/eticn/```

##### 2.2.2 stf为基座模型
* **qeevrf**：```python example/eval_vbr.py --model_config_path config/stf/stf_qeevrf_stage3.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_qeevrf/stage3 --save_path result/R-D/vbr/stf/[Ours].json```
* **qvrf**：```python example/eval_vbr.py --model_config_path config/stf/stf_qvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_qvrf/stage2 --save_path result/R-D/vbr/stf/[Tong(NIPS2023)].json```
* **vgvrf**：```python example/eval_vbr.py --model_config_path config/stf/stf_vgvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_vgvrf/ --save_path result/R-D/vbr/stf/[Cai(CVPR2022)].json```
* **stvrf**：```python example/eval_vbr.py --model_config_path config/stf/stf_stvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_stvrf/ --save_path result/R-D/vbr/stf/[Presta(TIP2025)].json```
* **msd**：```python example/eval_vbr.py --model_config_path config/stf/stf_msd.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_msd/ --save_path result/R-D/vbr/stf/camvid/[MSD].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/stf --base_result_file Baseline[Zou(CVPR2022)].json```
* RD曲线：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/stf/```

##### 2.2.3 gric为基座模型
* **qeevrf**：```python example/eval_vbr.py --model_config_path config/gric/gric_qeevrf_stage3.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/gric/gric_qeevrf/stage3 --save_path result/R-D/vbr/gric/[Ours].json```
* **qvrf**：```python example/eval_vbr.py --model_config_path config/gric/gric_qvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/gric/gric_qvrf/stage2 --save_path result/R-D/vbr/gric/[Tong(NIPS2023)].json```
* **vgvrf**：```python example/eval_vbr.py --model_config_path config/gric/gric_vgvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/gric/gric_vgvrf/ --save_path result/R-D/vbr/gric/[Cai(CVPR2022)].json```
* **stvrf**：```python example/eval_vbr.py --model_config_path config/gric/gric_stvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/gric/gric_stvrf/ --save_path result/R-D/vbr/gric/[Presta(TIP2025)].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/gric --base_result_file Baseline[Yang(NIPS2024)].json```
* RD曲线：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/gric/```

##### 2.2.4 vaic为基座模型
**qeevrf**：```python example/eval_vbr.py --model_config_path config/vaic/vaic_qeevrf_stage3.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vaic/vaic_qeevrf/stage3 --save_path result/R-D/vbr/vaic/camvid/[Ours].json```
**qvrf**：```python example/eval_vbr.py --model_config_path config/vaic/vaic_qvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vaic/vaic_qvrf/stage2 --save_path result/R-D/vbr/vaic/camvid/[Tong(NIPS2023)].json```
**vgvrf**：```python example/eval_vbr.py --model_config_path config/vaic/vaic_vgvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vaic/vaic_vgvrf/ --save_path result/R-D/vbr/vaic/camvid/[Cai(CVPR2022)].json```
**stvrf**：```python example/eval_vbr.py --model_config_path config/vaic/vaic_stvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vaic/vaic_stvrf/ --save_path result/R-D/vbr/vaic/camvid/[Presta(TIP2025)].json```
**msd**：```python example/eval_vbr.py --model_config_path config/vaic/vaic_msd.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vaic/vaic_msd/ --save_path result/R-D/vbr/vaic/camvid/[MSD].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/vaic --base_result_file Baseline[Minnen(NIPS2018)].json```
* RD曲线：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/vaic/```

#### 2.2.2 测试集

##### 2.2.2.1 soda

* **qeevrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qeevrf_stage3.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/eticn/eticn_qeevrf/stage3 --save_path result/R-D/vbr/soda/eticn/[Ours].json```
* **qvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qvrf.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/eticn/eticn_qvrf/stage2 --save_path result/R-D/vbr/soda/eticn/[Tong(NIPS2023)].json```
* **vgvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_vgvrf.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/eticn/eticn_vgvrf/ --save_path result/R-D/vbr/soda/eticn/[Cai(CVPR2022)].json```
* **stvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_stvrf.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/eticn/eticn_stvrf/ --save_path result/R-D/vbr/soda/eticn/[Presta(TIP2025)].json```
* **msd**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_msd.yml --data_path datasets/soda/soda_train/val.jsonl --model_path saved_model/eticn/eticn_msd/ --save_path result/R-D/vbr/soda/eticn/[MSD].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/soda/eticn --base_result_file Baseline[ETICN].json```
* RD结果：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/soda/eticn```

##### 2.2.2.2 srti

* **qeevrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qeevrf_stage3.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/eticn/eticn_qeevrf/stage3 --save_path result/R-D/vbr/srti/eticn/[Ours].json```
* **qvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qvrf.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/eticn/eticn_qvrf/stage2 --save_path result/R-D/vbr/srti/eticn/[Tong(NIPS2023)].json```
* **vgvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_vgvrf.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/eticn/eticn_vgvrf/ --save_path result/R-D/vbr/srti/eticn/[Cai(CVPR2022)].json```
* **stvrf**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_stvrf.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/eticn/eticn_stvrf/ --save_path result/R-D/vbr/srti/eticn/[Presta(TIP2025)].json```
* **msd**：```python example/eval_vbr.py --model_config_path config/eticn/eticn_msd.yml --data_path datasets/srti/srti_train/val.jsonl --model_path saved_model/eticn/eticn_msd/ --save_path result/R-D/vbr/srti/eticn/[MSD].json```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/srti/eticn --base_result_file Baseline[ETICN].json```
* RD结果：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/srti/eticn/```

### 2.3 消融实验

#### 2.3.1 消融1
* eticn: ```python example/eval.py --model_config_path config/eticn/eticn.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/ --save_path result/R-D/fbr/ablation/e1/ETICN.json```
* neta：```python example/eval.py --model_config_path config/neta/neta_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/neta/ --save_path result/R-D/fbr/ablation/e1/NET-A.json```
* netb：```python example/eval.py --model_config_path config/stf/stf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/ --save_path result/R-D/fbr/ablation/e1/NET-B.json```
* netc：```python example/eval.py --model_config_path config/netc/netc_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/netc/ --save_path result/R-D/fbr/ablation/e1/NET-C.json```
* R-D曲线：```python example/plot_R_D_curve.py --dir_path result/R-D/fbr/ablation/e1/```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/fbr/ablation/e1/ --base_result_file NET-C.json```

#### 2.3.2 消融2
##### 2.3.2.1 G = 32

* **G = 32, C = 256**：```python example/eval.py --model_config_path config/ablation/eticn_256_32/eticn_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn_256_32/ --save_path result/R-D/fbr/ablation/e2/G=32/C=256.json```
* **G = 32, C = 512**：```python example/eval.py --model_config_path config/ablation/eticn_512_32/eticn_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn_512_32/ --save_path result/R-D/fbr/ablation/e2/G=32/C=512.json```
* **G = 32, C = 1024**：```python example/eval.py --model_config_path config/ablation/eticn_1024_32/eticn_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn_1024_32/ --save_path result/R-D/fbr/ablation/e2/G=32/C=1024.json```
* **R-D曲线**：```python example/plot_R_D_curve.py --dir_path result/R-D/fbr/ablation/e2/G=32/```
* **BD结果**：```python example/get_bp_psnr_rate.py --dir_path result/R-D/fbr/ablation/e2/G=32/ --base_result_file Dim=256.json```

##### 2.3.2.2 C = 512
* **G = 16, C = 512**：```python example/eval.py --model_config_path config/ablation/eticn_512_16/eticn_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn_512_16/ --save_path result/R-D/fbr/ablation/e2/C=512/G=16.json```
* **G = 32, C = 512**：```python example/eval.py --model_config_path config/ablation/eticn_512_32/eticn_0018.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn_512_32/ --save_path result/R-D/fbr/ablation/e2/C=512/G=32.json```
* **R-D曲线**：```python example/plot_R_D_curve.py --dir_path result/R-D/fbr/ablation/e2/C=512/```
* **BD结果**：```python example/get_bp_psnr_rate.py --dir_path result/R-D/fbr/ablation/e2/C=512/ --base_result_file Group=16.json```

#### 2.3.3 消融3
* qeevrf:```python example/eval_vbr.py --model_config_path config/eticn/eticn_qeevrf_stage3.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_qeevrf/stage3 --save_path result/R-D/vbr/ablation/QEEVRF.json```
* neta：```python example/eval_vbr.py --model_config_path config/eticn/eticn_qeevrf_stage2.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_qeevrf/stage2 --save_path result/R-D/vbr/ablation/NET-A.json```
* netb:```python example/eval_vbr.py --model_config_path config/eticn/eticn_qvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/eticn/eticn_qvrf/stage2 --save_path result/R-D/vbr/ablation/NET-B.json```
* R-D曲线：```python example/plot_R_D_curve.py --dir_path result/R-D/vbr/ablation/```
* BD结果：```python example/get_bp_psnr_rate.py --dir_path result/R-D/vbr/ablation/ --base_result_file NET-B.json```





* neta：```python example/eval_vbr.py --model_config_path config/stf/stf_qeevrf_stage2.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_qeevrf/stage2 --save_path result/R-D/vbr/ablation/NET-A.json```
* netb:```python example/eval_vbr.py --model_config_path config/stf/stf_qvrf.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/stf/stf_qvrf/stage2 --save_path result/R-D/vbr/ablation/NET-B.json``



