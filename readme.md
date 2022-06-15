This is the data and code for paper
[Robust Textual Embedding against Word-level Adversarial Attacks (UAI 2022)](https://arxiv.org/abs/2202.13817).

## Datesets
There are three datasets used in our experiments. Download and uncompress them to the directory  `./data/`. Rename their paths to `./data/imdb`, `./data/yelp/` and `./data/yahoo/`, respectively.

- [IMDB](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
- [Yelp-5](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz)
- [Yahoo! Answers](https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz)

## Dependencies
Download and put `glove.840B.300d.txt`, `counter-fitted-vectors.txt`, `pytorch_model.bin`, and `bert_config.json` to the directory `./data/`.

- [GloVe vecors](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- [Counter fitted vectors](https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip)
- [Pre-trained BERT](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)

## Requirements
- python==3.7.11
- pytorch==1.7.1
- tensorflow-gpu==1.15.0
- tqdm==4.42
- scikit-learn==0.23
- numpy==1.21
- keras==2.2.5
- nltk==3.4.5

## Experiments

### For CNN/BiLSTM models:

1. Generating the dictionary, embedding matrix and distance matrix:

    ```shell
    python build_embs.py --task_name imdb --data_dir ./data/
    ```

    Depending on the dataset you want to use, the `--task_name` field can be `imdb`, `yelp`, or `yahoo`.

    You could use our pregenerated data by downloading [aux_files](https://drive.google.com/file/d/1lh5gMVkDEqKjoZD1beXtjya5QgY6Pvmq/view?usp=sharing) and place `aux_files` into the dictionary `./data/`.

2. Training the model with standard trainig:

    ```shell
    python cnn_classifier.py --data_dir ./data/ --task_name imdb --model_type CNNModel --output_dir model/cnn-imdb-nt --do_train --do_eval --max_seq_length 512 --num_train_epochs 2
    ```

    Depending on the model you want to use, the `--model_type` field can be `CNNModel` or `BiLSTMModel`. The `--max_seq_length` is `512` for `imdb` and `256` for `yelp` and `yahoo`.

3. Attacking the model of stardard training using the attack GA:

    ```shell
    python cnn_attack.py --data_dir ./data/ --task_name imdb --model_type CNNModel --attack ga --output_dir model/cnn-imdb-nt --save_to_file model/cnn-imdb-nt/attack-ga-2.txt --max_seq_length 512 --num_train_epochs 2
    ```

    Depending on the attack method you want to use, the `--attack` field can be `pwws`, `ga` or `pso`.

4. Training the model with our proposed FTML:
    
    ```shell
    python cnn_classifier_ftml.py --data_dir ./data/ --task_name imdb --model_type CNNModel --output_dir model/cnn-imdb-ftml --do_train --do_eval --max_seq_length 512 --num_train_epochs 20 --beta 1.0 --alpha 6.0
    ```

    The `--num_train_epochs` is `20` for `imdb` and `5` for `yelp` and `yahoo`.

    You could also use our trained model by downloading [models](https://drive.google.com/file/d/1ackInH0I-wfLwZxfsYrslbhYQlKLYX6o/view?usp=sharing).

5. Attacking the model of FTML using the attack GA:

    ```shell
    python cnn_attack.py --data_dir ./data/ --task_name imdb --model_type CNNModel --attack ga --output_dir model/cnn-imdb-ftml --save_to_file model/cnn-imdb-ftml/attack-ga-20.txt --max_seq_length 512 --num_train_epochs 20
    ```

### For BERT models:

1. Training the model with standard trainig:

    ```shell
    python bert_classifier.py --data_dir ./data/ --task_name imdb --output_dir model/bert-imdb-nt --max_seq_length 256 --do_train --do_eval --do_lower_case --num_train_epochs 3
    ```

2. Attacking the model of stardard training using the attack GA:

    ```shell
    python bert_attack.py --data_dir ./data/ --task_name imdb --attack ga --output_dir model/bert-imdb-nt --save_to_file model/bert-imdb-nt/attack-ga-3.txt --do_lower_case --num_train_epochs 3
    ```

3. Training the model with our proposed FTML:
    
    ```shell
    python bert_classifier_ftml.py --data_dir ./data/ --task_name imdb --output_dir model/bert-imdb-ftml --do_train --do_eval --do_lower_case --num_train_epochs 20
    ```
    You could also use our trained model by downloading [models](https://drive.google.com/file/d/1ackInH0I-wfLwZxfsYrslbhYQlKLYX6o/view?usp=sharing).

4. Attacking the model of FTML using the attack GA:
    
    ```shell
    python bert_attack.py --data_dir ./data/ --task_name imdb --attack ga --output_dir model/bert-imdb-ftml --save_to_file model/bert-imdb-ftml/attack-ga-20.txt --do_lower_case --num_train_epochs 20
    ```