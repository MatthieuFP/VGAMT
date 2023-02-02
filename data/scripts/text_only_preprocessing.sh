#!/bin/bash

# Export moses path
PATH=mosesdecoder/scripts/tokenizer:${PATH}
PATH=mosesdecoder/scripts/training:${PATH}

REPLACE_UNICODE_PUNCT=replace-unicode-punctuation.perl
NORM_PUNC=normalize-punctuation.perl
REM_NON_PRINT_CHAR=remove-non-printing-char.perl
CLEAN_CORPUS=clean-corpus-n.perl
SRC="$1"
TGT="$2"


DATA_PATH="PATH/TO/TEXT-ONLY/DATA/${SRC}-${TGT}"
dataset=(opensubtitles wikipedia ted2020 ted2013 books)


for data_set in "${dataset[@]}"; do
  echo ${data_set}
  dir_name="${DATA_PATH}/${data_set}"
  src_raw_file="${dir_name}/*${SRC}"
  tgt_raw_file="${dir_name}/*{TGT}"
  for file in ${src_raw_file} ${tgt_raw_file}; do
    fname=`basename ${file}`
    lg=${fname: -2}
    cat "${file}" | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR > ${dir_name}/normalized.${fname} &
  done
  wait
  $CLEAN_CORPUS ${dir_name}/normalized.${fname::-3} ${SRC} ${TGT} ${dir_name}/clean 4 100
done
wait