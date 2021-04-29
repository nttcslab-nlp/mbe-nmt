#!/bin/sh

SRC=en
TRG=ja

MAX_LEN=250

# # Download
mkdir -p orig
wget -P orig http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/2.0/bitext/en-ja.tar.gz
wget -P orig http://data.statmt.org/wmt20/translation-task/dev.tgz ./orig
# wget -P orig https://wit3.fbk.eu/archive/2017-01-trnted//texts/en/ja/en-ja.tgz ./orig
mkdir -p orig/jparacrawl orig/wmt orig/iwslt
tar xzvf orig/en-ja.tar.gz -C orig/jparacrawl
tar xzvf orig/dev.tgz -C orig/wmt
tar xzvf orig/en-ja.tgz -C orig/iwslt

# Raw
mkdir -p raw
sort orig/jparacrawl/en-ja/en-ja.bicleaner05.txt > orig/jparacrawl/en-ja/en-ja.sort
cut -f 1 orig/jparacrawl/en-ja/en-ja.sort > raw/jparacrawl.doc
cut -f 3 orig/jparacrawl/en-ja/en-ja.sort > raw/jparacrawl.en
cut -f 4 orig/jparacrawl/en-ja/en-ja.sort > raw/jparacrawl.ja
cat orig/iwslt/en-ja/IWSLT17.TED.tst2012.en-ja.en.xml | ../scripts/input-from-sgm.perl > raw/tst2012.en
cat orig/iwslt/en-ja/IWSLT17.TED.tst2012.en-ja.ja.xml | ../scripts/input-from-sgm.perl > raw/tst2012.ja
python ../scripts/sgm_to_doc.py orig/iwslt/en-ja/IWSLT17.TED.tst2012.en-ja.en.xml > raw/tst2012.doc
cat orig/wmt/dev/newsdev2020-enja-src.en.sgm | ../scripts/input-from-sgm.perl > raw/newsdev2020-enja.en
cat orig/wmt/dev/newsdev2020-enja-ref.ja.sgm | ../scripts/input-from-sgm.perl > raw/newsdev2020-enja.ja
python ../scripts/sgm_to_doc.py orig/wmt/dev/newsdev2020-enja-src.en.sgm > raw/newsdev2020-enja.doc

# SPM
mkdir -p detok
for MERGE in 32000; do
    mkdir -p spm/$MERGE/spm_model
    spm_train --input=./raw/jparacrawl.en --model_prefix=./spm/$MERGE/spm_model/spm.en.nopretok --vocab_size=$MERGE --character_coverage=1.0 --model_type=unigram --normalization_rule_name=nmt_nfkc &
    spm_train --input=./raw/jparacrawl.ja --model_prefix=./spm/$MERGE/spm_model/spm.ja.nopretok --vocab_size=$MERGE --character_coverage=1.0 --model_type=unigram --normalization_rule_name=nmt_nfkc &
    wait
    for L in $SRC $TRG; do
        for FILE in ./raw/*.$L; do
            BASE=`basename $FILE`
            spm_encode --model=./spm/$MERGE/spm_model/spm.$L.nopretok.model --output_format=piece < $FILE > spm/$MERGE/$BASE &
        done
    done
    wait
    for L in $SRC $TRG; do
        for FILE in ./raw/*.$L; do
            BASE=`basename $FILE`
            spm_decode --model=./spm/$MERGE/spm_model/spm.$L.nopretok.model < spm/$MERGE/$BASE > detok/$BASE &
        done
    done
    wait

    # Clean
    mkdir -p ./spm/$MERGE/clean
    ln -s $PWD/raw/jparacrawl.doc ./spm/$MERGE
    ../scripts/clean-corpus-n-doc.perl ./spm/$MERGE/jparacrawl $SRC $TRG doc ./spm/$MERGE/clean/jparacrawl 1 $MAX_LEN
done
