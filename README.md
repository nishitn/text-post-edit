# text-post-edit
Using Facebooks fairseq library for a project
Link to Original -  [fairseq github](https://github.com/pytorch/fairseq/)

# Pre Processing

Pre-Porcess binarises guess file now, no extra code 
file name format = train.target_lang.guess test.target_lang.guess valid.target_lang.guess
Preprocess Code Example
```
python3 preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14.tokenized.de-en
```

# Translate line-by-line
```translate.py``` translates line-by-line from your existing model and saves output in ```logs/translate.txt```
