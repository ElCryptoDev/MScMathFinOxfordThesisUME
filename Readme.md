

## Arguments of command line interface:

```
usage: task.py [-h] --job-dir JOB_DIR --run-name RUN_NAME
               [--coin {BTC,ETH,XMR,XRP,LTC,BCH,XLM,DASH,IOTA,TRX,ndx,DJIA}]
               [--run_config RUN_CONFIG] [--sample {1m,10m,1h,1d}]
               [--data-path DATA_PATH]
               [--loss {mean_squared_error,mean_absolute_percentage_error,prediction_of_change_in_direction,prediction_of_change_in_direction_ret,u_theil,average_relative_variance,index_agreement,sum_of_losses_and_gains,sum_of_losses_and_gains_ret,binary_crossentropy}]
               [--num-layers NUM_LAYERS] [--num-nodes NUM_NODES]
               [--frac-dropout FRAC_DROPOUT] [--num-epochs NUM_EPOCHS]
               [--batch-size BATCH_SIZE] [--lookback LOOKBACK] [--step STEP]
               [--delay DELAY]
               [--steps-per-epoch-override STEPS_PER_EPOCH_OVERRIDE]
               [--temporal-attention TEMPORAL_ATTENTION]
               [--input-attention INPUT_ATTENTION]
               [--attention-style {luong,bahdanau}]
               [--verbosity {DEBUG,ERROR,FATAL,INFO,WARN}]
               [--validation-freq VALIDATION_FREQ]
               [--model_type {simple,darnn,self_attn}]
               [--learning-rate LEARNING_RATE] [--run-config RUN_CONFIG]
               [--categorical CATEGORICAL]

optional arguments:
  -h, --help            show this help message and exit
  --job-dir JOB_DIR     GCS location to write checkpoints and export models
  --run-name RUN_NAME   Name of experiment
  --coin {BTC,ETH,XMR,XRP,LTC,BCH,XLM,DASH,IOTA,TRX,ndx,DJIA}
                        Name of coin to perfrom training for.
  --run_config RUN_CONFIG
                        Path of yaml file for configuration of target and
                        features
  --sample {1m,10m,1h,1d}
                        Sample type to load. Available frequencies "10m", "1h"
                        and "1d"
  --data-path DATA_PATH
                        Location of training files (local or GCS e.g.
                        gs://sample_ai_bucket/)
  --loss {mean_squared_error,mean_absolute_percentage_error,prediction_of_change_in_direction,prediction_of_change_in_direction_ret,u_theil,average_relative_variance,index_agreement,sum_of_losses_and_gains,sum_of_losses_and_gains_ret,binary_crossentropy}
                        Loss function for training
  --num-layers NUM_LAYERS
                        number of layers in LSTM model, default=2
  --num-nodes NUM_NODES
                        number of layers in LSTM model, default=128
  --frac-dropout FRAC_DROPOUT
                        dropout fraction, default=0
  --num-epochs NUM_EPOCHS
                        number of times to go through the data, default=5
  --batch-size BATCH_SIZE
                        number of records to read during each training step,
                        default=512
  --lookback LOOKBACK   number of lookback time steps in input sequence,
                        default=60
  --step STEP           step size for selection timesteps in input sequance,
                        default=1
  --delay DELAY         time difference of input series' end and prediction
                        timestep, default=0 (1 time step ahead)
  --steps-per-epoch-override STEPS_PER_EPOCH_OVERRIDE
                        override value for number of steps per epoch
  --temporal-attention TEMPORAL_ATTENTION
                        Add temporal attention layer on bottom of LSTM
                        network.
  --input-attention INPUT_ATTENTION
                        Add input attention layer on top of LSTM network.
  --attention-style {luong,bahdanau}
                        Configure attention stlye: 'luong' or 'bahdanau')
  --verbosity {DEBUG,ERROR,FATAL,INFO,WARN}
  --validation-freq VALIDATION_FREQ
                        frequency to carry out validation steps in times of
                        epochs.
  --model_type {simple,darnn,self_attn}
                        Selector for model architecture.
  --learning-rate LEARNING_RATE
                        Learning rate for Adam optimizer.
  --run-config RUN_CONFIG
                        Location of yaml parameter configuration file.
  --categorical CATEGORICAL
                        Type of prediction: categorical vs. regression.
```


## How to run the code:

Install gcloud sdk. Run code locally:

```
gcloud ai-platform local train --package-path=trainer --module-name trainer.task --job-dir <job-dir> -- further arguments
```

Run code on Google Cloud Platform via ai-platform API:

```
gcloud ai-platform jobs submit training run_name --package-path=trainer --module-name trainer.task --job-dir gs://<bucket_name> --region us-central1 --scale-tier BASIC_GPU --runtime-version 2.1 --python-version 3.7 -- further arguments
```

