# MFTCN: A Multi-Frequency Temporal Convolutional Network for Short-Term Temperature Prediction
Daily mean temperature data from March 2010 to March 2020 of Wushan and Youyang in Chongqing, and main codes of the model.

Take temperature data from Wushan, we used historical temperature series of different lengths to predict future short-term air temperature. ARIMA, XGBoost, LSTM, TCN, EEMD-LSTM and EEMD-TCN are compared with our model, the proposed model showed the best performance. 

The experimental configuration: Ubuntu 16.04.7, Intel(R) Xeon(R) CPU E5620 @ 2.40GHz, GeForce GTX 1080 Ti, Python 3.7.9, Pytorch 1.7.1.


**If you use this code in your research, please cite our work:**  
```bibtex
@INPROCEEDINGS{10727066,
  author={Huang, Wei and Gao, Yuan and Fu, Xiangling and Song, Zhiyi},
  booktitle={2024 IEEE 24th International Conference on Software Quality, Reliability, and Security Companion (QRS-C)}, 
  title={MFTCN: A Multi-Frequency Temporal Convolutional Network for Enhancing Temperature Prediction Dependability}, 
  year={2024},
  pages={279-288},
  keywords={Temperature distribution;Temperature dependence;Accuracy;Uncertainty;Biological system modeling;Predictive models;Data models;Complexity theory;Convolutional neural networks;Optimization;Multi-frequency decomposition;Temporal convolutional network;Short-term temperature prediction},
  doi={10.1109/QRS-C63300.2024.00044}}
```
