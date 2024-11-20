import torch
import os
import logging
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size

logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    encoder = setup_encoder(train_config, model_config, **kwargs)
    llm = setup_llm(train_config, model_config, **kwargs)
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    
    model = slam_model_asr(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer

class slam_model_asr(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )
        self.dataset_config = kwargs.get("dataset_config", None)

    @torch.no_grad()
    def inference(
        self,
        wav_path=None,
        **kwargs,
    ):
        device = kwargs.get("device", "cuda")
        
        # Get the experiment type
        exp_type = getattr(self.train_config, "experiment_type", None)
        if exp_type not in ["exp1", "exp2", "exp3"]:
            raise ValueError(f"Invalid or no experiment type specified in train_config: {exp_type}")

        # Processing audio features
        encoder_outs = None
        transcribed_text = ""
        if os.path.exists(wav_path) and exp_type in ["exp1", "exp2"]:
            import whisper
            
            # 1. Loading and preprocessing the adio
            audio_raw = whisper.load_audio(wav_path)
            audio_raw = whisper.pad_or_trim(audio_raw)
            
            # 2. extract audio features
            mel_size = getattr(self.dataset_config, "mel_size", 80)
            audio_mel = (
                whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                .permute(1, 0)[None, :, :]
                .to(device)
            )
            
            # 3. Get the whisper transcription
            try:
                whisper_model = whisper.load_model("base")
                transcription_result = whisper_model.transcribe(wav_path)
                transcribed_text = transcription_result["text"]
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}")
                transcribed_text = "[Transcription failed]"
            
            # 4. Process audio features
            encoder_outs = self.encoder.extract_features(
                audio_mel.permute(0, 2, 1)
            )[0]
            
            if self.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            elif self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if encoder_outs is None:
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        # Get the prompt template
        base_prompt = getattr(self.dataset_config, "prompt", "")
        
        # Build the prompt based on experiment type
        if exp_type in ["exp2", "exp3"]:
            prompt = f"""USER: The transcription of the audio is: "{transcribed_text}"

{base_prompt}

A:"""
        else:  # exp1
            prompt = f"USER: {transcribed_text}\nASSISTANT:"

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)
        
        # Get the token embeddings
        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)
        
        # Use of audio features only at exp1 and exp2
        if exp_type in ["exp1", "exp2"]:
            inputs_embeds = torch.cat(
                (encoder_outs, inputs_embeds[None, :, :]), dim=1
            )
        else:
            inputs_embeds = inputs_embeds[None, :, :]
        
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(device)

        generation_kwargs = {
            'max_new_tokens': 10,          # Limit output length
            'min_new_tokens': 2,           # Ensure complete answers
            'temperature': 0.7,            # Reduced randomness
            'do_sample': False,            # Using Greedy Decoding
            'num_beams': 1,               # Simple Beam Search
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        generation_kwargs.update(kwargs)  # Allow overriding of default values

        # Generate and clean up output
        raw_output = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        decoded_output = self.tokenizer.decode(raw_output[0], skip_special_tokens=True)
        return self.clean_output_text(decoded_output)

    def clean_output_text(self, text):
        """
        Clean and standardize model output
        Args:
            text (str): Raw output text from model
        Returns:
            str: Cleaned text containing only 'Yes' or 'No'
        """
        # Remove common artifacts
        text = text.split('$')[0]  # Remove everything after '$'
        text = text.split('###')[0]  # Remove everything after '###'
        text = text.split('Question:')[0]  # Remove Q&A format
        text = ''.join(char for char in text if ord(char) < 128)  # Remove non-ASCII chars
        
        # Clean and standardize
        text = text.strip().lower()
        
        # Strict matching for Yes/No
        if 'yes' in text[:10]:  # Only check the beginning
            return 'Yes'
        elif 'no' in text[:10]:
            return 'No'
        
        # Default response
        return 'No'
