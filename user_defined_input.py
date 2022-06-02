import prepare_data
import pickle
import argparse
import utils
import torch
import numpy as np
import copy

from pathlib import Path
from train import XLNetForPredictingMiddleNotes
from transformers import XLNetConfig


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

configuration = XLNetConfig().from_dict({
  "_name_or_path": "xlnet-predict-middle-notes",
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": False,
  "bos_token_id": 10000,
  "clamp_len": -1,
  # "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": None, # null
  "model_type": "xlnet",
  "n_head": 8,  # 12 originally
  "n_layer": 12,
  "pad_token_id": 10000,
  "reuse_len": None, # null,
  "same_length": False,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": True,
  "untie_r": True,
  "use_mems_eval": True,
  "use_mems_train": True,
  # "vocab_size": 32000
})

def convert_midis_to_worded_data(midi1, midi2):
    midis = []
    try:
        note_items, tempo_items = utils.read_items(midi1)
        midis.append(midi1)
        note_items, tempo_items = utils.read_items(midi2)
        midis.append(midi2)
    except Exception as e:
        print('At least one of the midi files is corrupted.', e)

    tuple_events = prepare_data.load_tuple_event(midis)
    save_data_path = 'worded_data_for_prediction.pickle'
    print(midis)

    prepare_data.tuple_event_to_word(tuple_events, dict_file=args.dict_file, save_path=save_data_path)

def prepare_data_for_prediction(data_file, e2w=None, w2e=None, length=0):

    assert e2w != None and w2e != None

    with open(data_file, 'rb') as handle:
        data = pickle.load(handle)

    assert len(data) == 2

    midi = data[0] + [[[0,0,0,0,0,0]] for _ in range(length)] + data[1]

    #set in both MIDI files the bar numbers in the worded data
    #for midi in data:
    for i in range(len(midi)):
        for note_tuple in midi[i]:
            note_tuple[1] = i

        # midis.append([copy.deepcopy(note_tuple) for bar in midi for note_tuple in bar])

    midis = np.array([[copy.deepcopy(note_tuple) for bar in midi for note_tuple in bar]])

    return midis, len(data[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    # training setup
    parser.add_argument('--dict-file', type=str, default='dictionary.pickle')
    parser.add_argument('--begin', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--for-dir', type=str)
    parser.add_argument('--length', type=int, default=4)
    parser.add_argument('--n-songs', type=int, default=1)

    # for prediction phase
    parser.add_argument('--ckpt-path', type=str, default="trained-model/loss34.ckpt", help='checkpoint to load.')

    args = parser.parse_args()
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    model = XLNetForPredictingMiddleNotes(configuration, e2w, w2e, is_train=False).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    if args.for_dir:
        for path1 in Path(args.for_dir).glob('*.mid'):
            for path2 in Path(args.for_dir).glob('*.mid'):
                if path1 != path2:
                    convert_midis_to_worded_data(path1, path2)
                    for i in range (1, 5):
                        data, gap_pos = prepare_data_for_prediction("worded_data_for_prediction.pickle", e2w=e2w, w2e=w2e, length=i)
                        print(f"Predict a midi file with begin: '{path1.stem}' and end: '{path2.stem}' starting prediction after bar: '{gap_pos}' for '{i}' bars ...")
                        model.user_defined_predict(data=data, n_songs=args.n_songs, target_start=gap_pos, target_end=gap_pos + i, filename=f"{path1.stem}__{path2.stem}.mid")
    else:
        convert_midis_to_worded_data(args.begin, args.end)
        data, gap_pos = prepare_data_for_prediction("worded_data_for_prediction.pickle", e2w=e2w, w2e=w2e, length=args.length)

        print(f"Predict a midi file with begin: '{args.begin}' and end: '{args.end}' starting prediction after bar: '{gap_pos}' for '{args.length}' bars ...")
        model.user_defined_predict(data=data, n_songs=args.n_songs, target_start=gap_pos, target_end=gap_pos + args.length, filename=f"{Path(args.begin).stem}__{args.length}__{Path(args.end).stem}.mid")

    torch.cuda.empty_cache()
