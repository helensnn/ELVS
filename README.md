 - [Introduction](#Introduction)
 - [Download VocalSound](#Download-VocalSound)
 - [Contact](#Contact)

## Introduction

You can train the model with the following command:

```
python run.py --lr 1e-4 --b 100 --n_class 6 --n-epochs 100 \
--freqm 48 --timem 192 --mixup 0 \
--data-train tr.json --data-val val.json --label-csv class_labels_indices_vs.csv --exp-dir exp/vocalsoundexp_dir \
--model eff_mean --model_size 1 --imagenet_pretrain False --save_model True \
--n-print-steps 100 --num-workers 16 --eval-dataset all  --eval-dir /datafiles --work_path /src
```

## Download-VocalSound

 [**VocalSound 16kHz Version** (1.7 GB, used in our experiment)](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1)

```
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```

## Contact
If you have a question, please bring up an issue (preferred) or send me an email helensnn@hotmail.com