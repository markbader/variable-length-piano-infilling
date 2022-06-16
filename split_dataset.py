from miditoolkit.midi import parser as mid_parser
from pathlib import Path
from argparse import ArgumentParser

def collect_filenames(read_dir: Path) -> list:
    # collect filenames of all .mid or .midi files in the given directory
    midi_files = []
    for ext in ("*.mid", "*.midi"):
        midi_files.extend(read_dir.rglob(ext))

    print("number of midis:", len(midi_files))
    return midi_files

def split_midi(midi_path: Path, save_dir: str) -> None:
    try:
        # load midi file
        mido_obj = mid_parser.MidiFile(midi_path)
        bar_length = mido_obj.ticks_per_beat * 4
        max_bar = mido_obj.max_tick
        seq_length = bar_length * 16
        step_size = bar_length * 8

        for start in range(0, max_bar - bar_length + seq_length, step_size):
            interval = (start, start + step_size)

            # export
            mido_obj.dump(filename=save_dir / f'{midi_path.stem}_{start}.mid', segment=interval)
    except Exception as e:
        print(f'Skipped midi {midi_path} because: {e}')

if __name__=="__main__":
    parser = ArgumentParser(description='')

    parser.add_argument('--midi-folder', type=str, default='datasets/MidiFiles_Training', help="Folder containing the midi files.")
    parser.add_argument('--save-folder', type=str, default='./training_data', help="Folder to save sliced midi files.")

    args = parser.parse_args()

    midi_folder = Path(args.midi_folder)
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    midis = collect_filenames(midi_folder)

    for path in midis:
        split_midi(path, save_folder)
