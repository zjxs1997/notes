class easy_field:
    def __init__(self, field):
        self.vocab = field.vocab
        self.init_token = field.init_token
        self.eos_token = field.eos_token
        self.pad_token = field.pad_token
        self.unk_token = field.unk_token
