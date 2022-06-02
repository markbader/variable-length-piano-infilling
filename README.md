# Variable-Length-Piano-Infilling
The official implementation of Variable-Length Piano Infilling (VLI). ([paper: Variable-Length Music Score Infilling via XLNet and Musically Specialized Positional Encoding](http://arxiv.org/abs/2108.05064))

VLI is a new Transformer-based model for music score infilling, i.e., to generate a polyphonic music sequence that fills in the gap between given past and future contexts. Our model can infill a variable number of notes for different time spans.

## Installation
1. Clone and install the modified [Huggingface Transformer](https://github.com/reichang182/Transformer) package.
```
pip install git+https://github.com/reichang182/Transformer.git
```
2. Clone this repo and install the required packages.
```
git clone https://github.com/reichang182/variable-length-piano-infilling.git
cd  variable-length-piano-infilling
pip install -r requirement.txt
```
3. Download and unzip the AIlabs.tw Pop1K7 dataset. (Download link: [here](https://drive.google.com/file/d/1qw_tVUntblIg4lW16vbpjLXVndkVtgDe/view?usp=sharing)).

## Training & Testing
	# Prepare data
	python3 prepare_data.py \
		--midi-folder datasets/midi/midi_synchronized/ \
		--save-folder ./

	# Train the model
	python3 train.py --train

	# Use the trained model for a prediction of a transition between two MIDI files
	python3 user_defined_input --begin <MIDI_FILE_1> --end <MIDI_FILE_2> --length <LENGTH>

	# Generate transitions from every file in the directory to every other file
	python3 user_defined_input --for-dir <PATH_TO_MIDI_FOLDER> --length <LENGTH>


## Architecture
<img src="figures/architecture.png" alt="drawing" width="600"/>

## Results
<figure style="background-color:red;">
  <img src="figures/training_loss.png" alt="drawing" width="600"/>
  <figcaption>The training NLL-loss curves of ours and the baseline models.</figcaption>
</figure>
<br><br><br>
<figure>
  <img src="figures/metric_difference.png" alt="drawing" width="600"/>
  <figcaption>The objective metrics evaluated on the music pieces generated by VLI(ours), ILM, FELIX, and the real music.</figcaption>
</figure>
<br><br><br>
<figure>
  <img src="figures/subjective_evaluation.png" alt="drawing" width="600"/>
  <figcaption>Results of the user study: mean opinion scores in 1–5 in M(melodic fluency), R(rhythmic fluency), I(im-pression), and percentage of votes in F(favorite), from ‘all’ the participants or only the music ‘pro’-fessionals.</figcaption>
</figure>
