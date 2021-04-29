#!/bin/bash

PROJECT_DIR=$PROJECT_DIR/project

FAIRSEQ=$PROJECT_DIR/tools/fairseq_doc

SEED=$1

EXP_NAME=big_baseline_seed$SEED

SRC=en
TRG=ja

TRAIN_SRC=$PROJECT_DIR/corpus/spm/32000/clean/jparacrawl.$SRC
TRAIN_TRG=$PROJECT_DIR/corpus/spm/32000/clean/jparacrawl.$TRG
TRAIN_DOC=$PROJECT_DIR/corpus/spm/32000/clean/jparacrawl.doc
DEV_SRC=$PROJECT_DIR/corpus/spm/32000/aspec_dev.$SRC
DEV_TRG=$PROJECT_DIR/corpus/spm/32000/aspec_dev.$TRG
DEV_DOC=$PROJECT_DIR/corpus/raw/aspec_dev.doc
TEST_SRC=$PROJECT_DIR/corpus/spm/32000/aspec_test.$SRC
TEST_TRG=$PROJECT_DIR/corpus/spm/32000/aspec_test.$TRG
TEST_DOC=$PROJECT_DIR/corpus/raw/aspec_test.doc
TEST_TRG_RAW=$PROJECT_DIR/corpus/detok/aspec_test.$TRG

SPM_MODEL=$PROJECT_DIR/corpus/spm/32000/spm_model/spm.$TRG.nopretok.model

CORPUS_DIR=$PWD/corpus
MODEL_DIR=$PWD/models/$EXP_NAME
DATA_DIR=$PWD/data-bin/$EXP_NAME

TRAIN_PREFIX=$CORPUS_DIR/$EXP_NAME/train
DEV_PREFIX=$CORPUS_DIR/$EXP_NAME/dev
TEST_PREFIX=$CORPUS_DIR/$EXP_NAME/test
DOC_PREFIX=$CORPUS_DIR/$EXP_NAME/doc

mkdir -p $CORPUS_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR

# make links to corpus
mkdir -p $CORPUS_DIR/$EXP_NAME
ln -s $TRAIN_SRC $TRAIN_PREFIX.$SRC
ln -s $TRAIN_TRG $TRAIN_PREFIX.$TRG
ln -s $TRAIN_DOC $DOC_PREFIX.train
ln -s $DEV_SRC $DEV_PREFIX.$SRC
ln -s $DEV_TRG $DEV_PREFIX.$TRG
ln -s $DEV_DOC $DOC_PREFIX.valid
ln -s $TEST_SRC $TEST_PREFIX.$SRC
ln -s $TEST_TRG $TEST_PREFIX.$TRG
ln -s $TEST_DOC $DOC_PREFIX.test

######################################
# Preprocessing
######################################
python $FAIRSEQ/preprocess.py \
    --source-lang $SRC \
    --target-lang $TRG \
    --trainpref $TRAIN_PREFIX \
    --validpref $DEV_PREFIX \
    --testpref $TEST_PREFIX \
    --destdir $DATA_DIR \
    --nwordssrc -1 \
    --nwordstgt -1 \
    --workers `nproc` \


######################################
# Training
######################################
python $FAIRSEQ/train.py $DATA_DIR \
    --arch transformer_wmt_en_de_big \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.001 \
    --min-lr 1e-09 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 8000 \
    --max-update 24000 \
    --save-dir $MODEL_DIR \
    --no-epoch-checkpoints \
    --save-interval 10000000000 \
    --validate-interval 1000000000 \
    --save-interval-updates 200 \
    --keep-interval-updates 8 \
    --log-format simple \
    --log-interval 5 \
    --ddp-backend no_c10d \
    --update-freq 5 \
    --fp16 \
    --seed $SEED \



######################################
# Averaging
######################################
rm -rf $MODEL_DIR/average
mkdir -p $MODEL_DIR/average
python $FAIRSEQ/scripts/average_checkpoints.py --inputs $MODEL_DIR --output $MODEL_DIR/average/average.pt --num-update-checkpoints 8


######################################
# Generate
######################################
# decode
B=`basename $TEST_SRC`

python $FAIRSEQ/generate.py $DATA_DIR \
    --gen-subset test \
    --path $MODEL_DIR/average/average.pt \
    --max-tokens 2000 \
    --beam 6 \
    --lenpen 1.0 \
    --log-format simple \
    --remove-bpe \
    | tee $MODEL_DIR/$B.hyp

grep "^H" $MODEL_DIR/$B.hyp | sed 's/^H-//g' | sort -n | cut -f3 > $MODEL_DIR/$B.true
cat $MODEL_DIR/$B.true | spm_decode --model=$SPM_MODEL --input_format=piece > $MODEL_DIR/$B.true.detok

cat $MODEL_DIR/$B.true.detok | sacrebleu --tokenize=ja-mecab $TEST_TRG_RAW | tee -a $MODEL_DIR/test.log
