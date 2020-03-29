# How to use it

1. Process raw data
```shell script
cd ${PROJECT_ROOT}
export PYTHONPATH = `pwd`/src
python -m gcmc.data mlog --rating=${RATING_DAT_FILE_PATH} \
    --user-feature=${USER_FEATURE_DAT_FILE_PATH} \
    --item-feature=${ITEM_FEATURE_DAT_FILE_PATH} \
    --valid_ratio=${VALID_RATIO} 
    --test_ratio=${TEST_RATIO}
    --dump=${FOLDER_TO_SAVE_DATASET}
```

2. Modify configuration file `${PROJECT_ROOT}/config/config_example.yml`.

3. Run configuration
```shell script
cd ${PROJECT_ROOT}
export PYTHONPATH = `pwd`/src
python -m gcmc.train -c config/config_example.yml 
```

## Caution

This code is just to demonstrate how to use the new dgl sampling API and how to train the GCMC model with multi GPU in 
a memory efficient manner. The hyper parameter in the configuration example file is not optimized. And the logic of
model validation, storage is not yet implemented.