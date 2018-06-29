import sys
import torch
from torch.autograd import Variable
import time

from fairseq import options, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from interactive import main as mn



def main(args):
    print(args)
    start_time=time.time()
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences, \
        '--max-sentences/--batch-size is not supported in interactive mode'

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    print('| loading model(s) from {}'.format(', '.join(args.path)))
    models, model_args = utils.load_ensemble_for_inference(args.path, data_dir=args.data)
    src_dict, dst_dict = models[0].src_dict, models[0].dst_dict

    print('| [{}] dictionary: {} types'.format(model_args.source_lang, len(src_dict)))
    print('| [{}] dictionary: {} types'.format(model_args.target_lang, len(dst_dict)))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        )

    # Initialize generator
    translator = SequenceGenerator(
        models, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen)
    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    i=0
    for src_str in file.readlines():
        src_str = src_str.strip()
        src_tokens = tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        if use_cuda:
            src_tokens = src_tokens.cuda()
        src_lengths = src_tokens.new([src_tokens.numel()])
        translations = translator.generate(
            Variable(src_tokens.view(1, -1)),
            Variable(src_lengths.view(-1)),
        )
        hypos = translations[0]

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                dst_dict=dst_dict,
                remove_bpe=args.remove_bpe,
            )
            y=hypo_str+'\n'
            target.write(y)
            i+=1
            elapsed_time=time.time()-start_time
            print('| Translating line ', i, ' Time Elapsed',elapsed_time,end='\r')
    
    elapsed_time=time.time()-start_time
    print('')
    print('| Translation done - Time Taken = ',elapsed_time)


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = parser.parse_args()
    input_file = input('| Type the Input file location and press return: ')
    file = open(input_file)
    target = open('logs/translate.txt', 'a') #-----------------------------------------------------------------------------------
    main(args) 
    file.close()
    target.close()
