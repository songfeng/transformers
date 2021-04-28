from ...configuration_utils import PretrainedConfig


COLBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/config.json",
    "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/config.json",
    "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/config.json",
    "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/config.json",
    "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/config.json",
    "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/config.json",
}


class ColBERTConfig(PretrainedConfig):
    model_type = "colbert"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        projection_dim: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type