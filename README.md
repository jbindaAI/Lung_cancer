# Benign/Malignant Classifier
Repository consists on code needed to build CNN classifier working on LIDC-IDRI dataset.
Solution is based on similar ConRad repository https://github.com/lenbrocki/ConRad developed by Lennart Brocki and Neo Christopher Chung.

Model is based on pretrained ResNet50 model and takes as input 32x32x32 crops from original LIDC scan volumes.

To run code:
1. Download LIDC-IDRI dataset from [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).
   I needed to download data and process it 100 hundred batches.
   Optionally, there is `LIDC_overview.ipynb` notebook with gentle introduction in LIDC dataset and pylidc library.
2. Run `Data_preparation.ipynb`. This notebook extract 32x32x32 crops from original LIDC-IDRI dataset. 
3. Run `FinetuneModel.ipynb`.

In weights folder there are weights for finetune models:
- Model1 -> weights for ResNet50 model trained on crops without mask application.
- Model2 -> weights for ResNet50 model trained on crops with mask application.


