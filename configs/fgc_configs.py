class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 5
        self.batch_size = 32
        self.train_batch_size = 32
        self.dev_batch_size = 256
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.weight_decay = 0.0
        self.lr = 1e-3
        self.epochs = 100
        self.early_stop_patience = 50
        self.scheduler_patience = 4
        self.scheduler_reduce_factor = 0.5
        self.optimizer = "AdamW"
        self.save_by = "accuracy"
        self.metric_direction = 1
        self.validation_interval = 1
        self.chunk_size = -1
        self.num_workers = 0
        self.display_metric = "accuracy"
        self.schedule = True
        self.custom_betas = False
        self.cache = True
        self.penalty_gamma = 0.0


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        # word embedding
        self.embd_dim = 64
        self.encoder_hidden_size = 128
        self.decoder_hidden_size = 128
        self.encoder_layers = 1
        self.decoder_layers = 1
        self.bidirectional = False
        self.heads = 1
        self.qk_head_dim = 128
        self.inter_value_size = 128
        self.v_head_dim = self.decoder_hidden_size
        self.position_max_len = 5000
        self.pointer = False
        self.cross_relative_pos = False
        self.encoder_type = "GRUEncoderDecoder"
        self.forward_positions = True
        self.reverse_positions = False
        self.mix_forward_reverse = False
        self.GRU_attention = False
        self.mix_attention = False
        self.location_attention_only = False
        self.dropout = 0.5
        self.attn_dropout = 0.0
        self.max_decoder_len = 500
        self.init_attn = 0.5
        self.softstair_temp = 20
        self.no_softstair = False
        self.position_past = False
        self.no_eos = False
        self.cheat_eos = False
        self.rope = False
        self.attention_type = "Multiheaded_Attention"


class BiGRU_config(base_config):
    def __init__(self):
        super().__init__()
        self.encoder_hidden_size = self.encoder_hidden_size // 2
        self.bidirectional = True
        self.model_name = "(BiGRU Seq2Seq)"

class BiGRUrel_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.forward_positions = True
        self.reverse_positions = False
        self.mix_forward_reverse = False
        self.cross_relative_pos = True
        self.model_name = "(BiGRU Rel Seq2Seq)"

class BiGRUrel_rev_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.forward_positions = False
        self.reverse_positions = True
        self.cross_relative_pos = True
        self.mix_forward_reverse = False
        self.model_name = "(BiGRU Rel Reverse Seq2Seq)"

class BiGRUrel_mixdir_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.reverse_positions = True
        self.forward_positions = True
        self.mix_forward_reverse = True
        self.cross_relative_pos = True
        self.model_name = "(BiGRU MixDir Seq2Seq)"


class BiGRUrope_config(BiGRUrel_config):
    def __init__(self):
        super().__init__()
        self.rope = True
        self.model_name = "(BiGRU ROPE Seq2Seq)"

class BiGRUrope_mixdir_config(BiGRUrel_mixdir_config):
    def __init__(self):
        super().__init__()
        self.rope = True
        self.model_name = "(BiGRU ROPE MixDir Seq2Seq)"

class BiGRU_locattn_simple_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = True
        self.location_attention_only = True
        self.mix_attention = False
        self.GRU_attention = False
        self.model_name = "(BiGRU locattn simple Seq2Seq)"

class BiGRU_locattn_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = False
        self.location_attention_only = True
        self.mix_attention = False
        self.GRU_attention = False
        self.model_name = "(BiGRU locattn Seq2Seq)"


class BiGRU_mixattn_simple_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = True
        self.location_attention_only = False
        self.mix_attention = True
        self.GRU_attention = False
        self.model_name = "(BiGRU mixattn simple Seq2Seq)"

class BiGRU_mixattn_simplePR_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = True
        self.location_attention_only = False
        self.mix_attention = True
        self.GRU_attention = False
        self.position_past = True
        self.model_name = "(BiGRU mixattn simple PR Seq2Seq)"


class BiGRU_mixattn_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = False
        self.location_attention_only = False
        self.mix_attention = True
        self.GRU_attention = False
        self.model_name = "(BiGRU mixattn Seq2Seq)"

class BiGRU_grumixattn_simple_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = True
        self.location_attention_only = False
        self.mix_attention = True
        self.GRU_attention = True
        self.model_name = "(BiGRU grumixattn simple Seq2Seq)"

class BiGRU_grumixattn_config(BiGRU_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_GRUMix_Attention"
        self.simplified = False
        self.location_attention_only = False
        self.mix_attention = True
        self.GRU_attention = True
        self.model_name = "(BiGRU grumixattn Seq2Seq)"

class BiGRU_OneStep_config(BiGRU_locattn_simple_config):
    def __init__(self):
        super().__init__()
        self.reverse_positions = True
        self.forward_positions = True
        self.mix_forward_reverse = True
        self.step_act = "sigmoid"
        self.attention_type = "Multiheaded_Monotonic_Attention"
        self.model_name = "(BiGRU OneStep Seq2Seq)"

class BiGRU_FreeStep_config(BiGRU_OneStep_config):
    def __init__(self):
        super().__init__()
        self.step_act = "free"
        self.model_name = "(BiGRU FreeStep Seq2Seq)"

class BiGRU_AblationStep_config(BiGRU_OneStep_config):
    def __init__(self):
        super().__init__()
        self.attention_type = "Multiheaded_Ablation_Attention"
        self.model_name = "(BiGRU OneStep Seq2Seq - Step 2)"

class BiGRU_SoftStairStep_config(BiGRU_OneStep_config):
    def __init__(self):
        super().__init__()
        self.step_act = "softstair"
        self.model_name = "(BiGRU SoftStairStep Seq2Seq)"


class BiGRU_MixOneStep_config(BiGRU_OneStep_config):
    def __init__(self):
        super().__init__()
        self.mix_attention = True
        self.model_name = "(BiGRU Mix OneStep Seq2Seq)"


class BiGRU_MixOneStepPR_config(BiGRU_MixOneStep_config):
    def __init__(self):
        super().__init__()
        self.position_past = True
        self.model_name = "(BiGRU Mix OneStep PR Seq2Seq)"

class BiGRU_MonoAttn_config(BiGRU_OneStep_config):
    def __init__(self):
        super().__init__()
        self.step_act = "controlled_relu"
        self.model_name = "(BiGRU MonoAttn Seq2Seq)"

class BiGRU_MixMonoAttn_config(BiGRU_MonoAttn_config):
    def __init__(self):
        super().__init__()
        self.mix_attention = True
        self.model_name = "(BiGRU Mix MonoAttn Seq2Seq)"


class BiGRU_MixMonoAttnPR_config(BiGRU_MixMonoAttn_config):
    def __init__(self):
        super().__init__()
        self.position_past = True
        self.model_name = "(BiGRU Mix MonoAttn PR Seq2Seq)"


class BiGRU_RMonoAttn_config(BiGRU_MonoAttn_config):
    def __init__(self):
        super().__init__()
        self.step_act = "relu"
        self.model_name = "(BiGRU RMonoAttn Seq2Seq)"

class BiGRU_MixRMonoAttn_config(BiGRU_RMonoAttn_config):
    def __init__(self):
        super().__init__()
        self.mix_attention = True
        self.model_name = "(BiGRU Mix RMonoAttn Seq2Seq)"

class BiGRU_MixRMonoAttnPR_config(BiGRU_MixRMonoAttn_config):
    def __init__(self):
        super().__init__()
        self.position_past = True
        self.model_name = "(BiGRU Mix RMonoAttn PR Seq2Seq)"
