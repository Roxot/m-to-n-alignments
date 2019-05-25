# From: github.com/Roxot/AEVNMT.pt
import argparse
import json

options = {
    # Format: "option_name": (type, default_val, required, description, group)
    # `group` is for ordering purposes when printing.

    # I/O and device information
    "hparams_file": (str, None, False, "A JSON file containing hyperparameter values.", 0),
    "training_prefix": (str, None, True, "The training file prefix.", 0),
    "validation_prefix": (str, None, True, "The validation file prefix.", 0),
    "vocab_prefix": (str, None, False, "The vocabulary prefix, if share_vocab is True"
                                       " this should be the vocabulary filename.", 0),
    "share_vocab": (bool, False, False, "Whether to share the vocabulary between the"
                                       " source and target language", 0),
    "src": (str, None, True, "The source language", 0),
    "tgt": (str, None, True, "The target language", 0),
    "use_gpu": (bool, False, False, "Whether to use the GPU or not", 0),
    "output_dir": (str, None, True, "Output directory.", 0),
    "subword_token": (str, None, False, "The subword token, e.g. \"@@\".", 0),
    "min_sentence_length": (int, 0, False, "The minimum sentence length during"
                                           " training.", 0),
    "max_sentence_length": (int, -1, False, "The maximum sentence length during"
                                            " training.", 0),
    "max_vocabulary_size": (int, -1, False, "The maximum vocabulary size.", 0),
    "vocab_min_freq": (int, 0, False, "The minimum frequency of a word for it"
                                      " to be included in the vocabulary.", 0),
    "model_checkpoint": (str, None, False, "A model checkpoint to load.", 0),

    # Model hyperparameters
    "model_type": (str, "neuralibm1", False, "The type of model to train:"
                                             " neuralibm1|bernoulli-RF|bernoulli-ST|hardkuma", 1),
    "cell_type": (str, "lstm", False, "The RNN cell type. rnn|gru|lstm", 1),
    "emb_size": (int, 32, False, "The source / target embedding size.", 1),
    "hidden_size": (int, 32, False, "The size of the hidden layers.", 1),
    "num_layers": (int, 1, False, "The number of encoder layers.", 1),
    "bidirectional": (bool, False, False, "Use a bidirectional encoder.", 1),
    "emb_init_scale": (float, 0.01, False, "Scale of the Gaussian that is used to"
                                           " initialize the embeddings.", 1),
    "pooling": (str, "avg", False, "Pooling to use: avg|sum", 1),
    "prior_param_1": (float, 0., False, "Prior parameter 1", 1),
    "prior_param_2": (float, 0., False, "Prior parameter 2", 1),

    # Optimization hyperparameters
    "num_epochs": (int, 1, False, "The number of epochs to train the model for.", 2),
    "learning_rate": (float, 1e-3, False, "The learning rate.", 2),
    "batch_size": (int, 64, False, "The batch size.", 2),
    "print_every": (int, 100, False, "Print training statistics every x steps.", 2),
    "max_gradient_norm": (float, -1.0, False, "The maximum gradient norm to clip the"
                                             " gradients to, to disable"
                                             " set <= 0.", 2),
    "lr_reduce_factor": (float, 0.5, False, "The factor to reduce the learning rate"
                                            " with if no validation improvement is"
                                            "  found.", 2),
    "lr_reduce_patience": (int, 2, False, "The number of evaluations to wait for"
                                           " improvement of validation scores"
                                           " before reducing the learning rate.", 2),
    "lr_reduce_cooldown": (int, 2, False, "The number of evaluations to wait with"
                                          " checking for improvements after a"
                                          " learning rate reduction.", 2),
    "min_lr": (float, 1e-5, False, "The minimum learning rate the learning rate"
                                   " scheduler can reduce to.", 2),
    "patience": (int, 5, False, "The number of evaluations to continue training for"
                                " when an improvement has been found.", 2),
    "dropout": (float, 0., False, "The amount of dropout.", 2),
    "evaluate_every": (int, -1, False, "The number of batches after which to run"
                                       " evaluation. If <= 0, evaluation will happen"
                                       " after every epoch.", 2),
    "KL_annealing_steps": (int, -1, False, "The number of steps to anneal the KL multiplier over,"
                                           " which goes from 0 to 1.", 2),
}

class Hyperparameters:

    """
        Loads hyperparameters from the command line arguments and optionally
        from a JSON file. Command line arguments overwrite those from the JSON file.
    """
    def __init__(self, check_required=True):
        self._hparams = {}
        self._defaulted_values = []
        cmd_line_hparams = self._load_from_command_line()

        # If given, load hparams from a json file.
        json_file = cmd_line_hparams["hparams_file"] if "hparams_file" in cmd_line_hparams \
                else None
        if json_file is not None:
            json_hparams = self._load_from_json(json_file)
            self._hparams.update(json_hparams)

        # Always override json hparams with command line hparams.
        self._hparams.update(cmd_line_hparams)

        # Set default values, check for required, set hparams as attributes.
        self._create_hparams(check_required)

    def update_from_file(self, json_file, override=False):
        json_hparams = self._load_from_json(json_file)
        if not override:
            for key, val in list(json_hparams.items()):
                if key not in self._defaulted_values:
                    del json_hparams[key]
                else:
                    self._defaulted_values.remove(key)
        self._hparams.update(json_hparams)
        self._create_hparams(False)

    def _load_from_command_line(self):
        parser = argparse.ArgumentParser()
        for option in options.keys():
            option_type, _, _, description, _ = options[option]
            option_type = str if option_type == bool else option_type
            parser.add_argument(f"--{option}", type=option_type,
                                help=description)
        args = parser.parse_args()
        cmd_line_hparams = vars(args)
        for key, val in list(cmd_line_hparams.items()):
            if val is None:
                del cmd_line_hparams[key]
        return cmd_line_hparams

    def _load_from_json(self, filename):
        with open(filename) as f:
            json_hparams = json.load(f)
        return json_hparams

    def _create_hparams(self, check_required):
        for option in options.keys():
            option_type, default_value, required, _, _ = options[option]
            if option not in self._hparams:
                self._hparams[option] = default_value
                self._defaulted_values.append(option)
            elif option_type == bool and isinstance(self._hparams[option], str):

                # Convert boolean inputs from string to bool. Only necessary if the
                # default value is not used.
                self._hparams[option] = str_to_bool(self._hparams[option])

            if self._hparams[option] is None:

                # Raise an error if required.
                if check_required and required:
                    raise Exception(f"Error: missing required value `{option}`.")

            setattr(self, option, self._hparams[option])

    def print_values(self):
        sorted_names = sorted(options.keys(), key=lambda name: (options[name][-1], name))
        cur_group = 0
        for name in sorted_names:
            if options[name][-1] > cur_group:
                print()
                cur_group = options[name][-1]

            val = self._hparams[name]
            defaulted = "(default)" if name in self._defaulted_values else ""
            print(f"{name} = {val} {defaulted}")

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self._hparams, f, sort_keys=True, indent=4)

def str_to_bool(string):
    return string.lower() == "true"
