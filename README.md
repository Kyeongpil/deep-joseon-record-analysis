# deep-joseon-record-analysis
Official implementation of restoration and translation of joseon historical records. (NAACL 2021)

https://www.aclweb.org/anthology/2021.naacl-main.317/

## Datasets and Model
You can download datasets and trained model weights via the following link:

https://tinyurl.com/yj7x8o2w

## Run codes
- First, you should preprocess the datasets
    
    ```
    python preprocessing.py 
    ```

- Then, you can train the model

    ```
    bash run_train.sh
    ```

- After training the model, you can translate the corpus using multi-gpus.

    The translation process have a master server and workers. 

    ```
    python translate_api.py
    ```

    ```
    (In the other shell) python translation_worker.py
    ```

- For the details of arguments, please refer codes.

## Dependencies
- Python 3.8+ (I recommend to install Anaconda Python.)
- PyTorch 1.8.1
- tensorboard
- sentencepiece
- mpi4py (conda install mpi4py)
- OpenMPI (conda install openmpi)
- ujson
- flask

## Contact
If you have any questions on our survey, please contact me via the following e-mail address: rudvlf0413@korea.ac.kr