import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from .Trans.transformers_encoder.transformer import TransformerEncoder
from .AlignMethods import AlignModule
from .inter_mamba import InterMamba as fusion_mamba


class mambafusion(nn.Module):
    def __init__(self, config, args):
        super(mambafusion, self).__init__()
        self.args = args

        self.linear_transform_visual = nn.Linear(256, 768)
        
        if self.args.need_aligned:

            self.alignNet = AlignModule(args, args.mag_aligned_method)
            
        self.fusion_method_final = fusion_mamba(dim=3, H=24, W=32, final=True)
        
    
    def forward(self, text_embedding, visual, acoustic):
        
        if self.args.need_aligned:
            text_embedding, visual, acoustic  = self.alignNet(text_embedding, visual, acoustic)

        
        fuse1 = self.fusion_method_final(text_embedding, visual)
        fuse2 = self.fusion_method_final(text_embedding, acoustic)
        
        
        return fuse1 + fuse2

        


class globalfusion(nn.Module):
    def __init__(self,  config, args):
        super(globalfusion, self).__init__()
        self.args = args

        if self.args.need_aligned:

            self.alignNet = AlignModule(args, args.mag_aligned_method)


        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        
        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)
        
        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)

        self.beta_shift = args.beta_shift

        self.LayerNorm = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        if self.args.need_aligned:
            text_embedding, visual, acoustic  = self.alignNet(text_embedding, visual, acoustic)
        
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output


class localfusion(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.gf = globalfusion(
            config, args
        )

        self.fusion_mamba = mambafusion(config, args)  
        self.args = args


        self.alignNet = AlignModule(args, args.aligned_method)

        self.embed_dim = args.text_feat_dim

        self.num_heads = args.nheads
   
        self.layers = args.n_levels
  
        self.attn_dropout = args.attn_dropout

        self.relu_dropout = args.relu_dropout

        self.res_dropout = args.res_dropout

        self.embed_dropout = args.embed_dropout

        self.attn_mask = args.attn_mask

        self.audio_proj = nn.Sequential(
            nn.LayerNorm(args.audio_feat_dim),
            nn.Linear(args.audio_feat_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.video_proj = nn.Sequential(
            nn.LayerNorm(args.video_feat_dim),
            nn.Linear(args.video_feat_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(args.text_feat_dim),
            nn.Linear(args.text_feat_dim, self.embed_dim),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, args.text_feat_dim)
        )

        self.trans_a_with_l = TransformerEncoder(embed_dim=self.embed_dim,
                                num_heads=self.num_heads,
                                layers=self.layers,
                                attn_dropout=self.attn_dropout,
                                relu_dropout=self.relu_dropout,
                                res_dropout=self.res_dropout,
                                embed_dropout=self.embed_dropout,
                                attn_mask=self.attn_mask)

        self.gamma = nn.Parameter(torch.ones(args.text_feat_dim) * 1e-4)


        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        condition_idx,
        ctx,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
    
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        batch_ctx = ctx.unsqueeze(0).repeat(acoustic.shape[0], 1, 1)

        _, aligned_visual, aligned_acoustic  = self.alignNet(batch_ctx, visual, acoustic)
        aligned_acoustic = self.audio_proj(aligned_acoustic)
        aligned_visual = self.video_proj(aligned_visual)
        batch_ctx = self.text_proj(batch_ctx)
        
        generated_ctx = self.fusion_mamba(batch_ctx, aligned_visual, aligned_acoustic)
                
        generated_ctx = batch_ctx + self.out_proj(generated_ctx) * self.gamma

        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = generated_ctx[i]


        fused_embedding = self.gf(embedding_output, visual, acoustic)
        

        conv_layer = nn.Conv2d(fused_embedding.shape[0], 1, kernel_size=1, bias=False).to(device)

        sigmoid_layer = nn.Sigmoid()
        mask = sigmoid_layer(conv_layer(fused_embedding))

        binary_mask = (mask > 0.50).float()
        fused_embedding = fused_embedding * binary_mask
        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
  
            encoder_hidden_states=encoder_hidden_states,

            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)


        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  
        return outputs, generated_ctx


class Local_Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.label_len = args.label_len
        self.bert = localfusion(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)

        self.init_weights()

    def forward(
        self,
        text,
        visual,
        acoustic,
        condition_idx,
        ctx,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]


        outputs, generated_ctx = self.bert(
            input_ids,
            visual,
            acoustic,
            condition_idx,
            ctx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        
        condition_tuple = tuple(sequence_output[torch.arange(sequence_output.shape[0]),\
            condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        
        condition = torch.cat(condition_tuple, dim=1)
        

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)


        outputs = (logits,) + outputs[
            2:
        ]  


        if labels is not None:
            if self.num_labels == 1:
 
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
        return outputs, pooled_output, condition, generated_ctx


class Cons_Model(BertPreTrainedModel):

    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.args = args

        self.post_init()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings


    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        condition_idx,
        ctx,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device


        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

       
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

       
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

       
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids,
      
            position_ids=position_ids,

            token_type_ids=token_type_ids,

            inputs_embeds=inputs_embeds,

            past_key_values_length=past_key_values_length,
        )

        

        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = ctx[i]


        encoder_outputs = self.encoder(
            embedding_output,
            
            attention_mask=extended_attention_mask,

            head_mask=head_mask,

            encoder_hidden_states=encoder_hidden_states,

            encoder_attention_mask=encoder_extended_attention_mask,

            past_key_values=past_key_values,

            use_cache=use_cache,

            output_attentions=output_attentions,

            output_hidden_states=output_hidden_states,

            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None


        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]


        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,

            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class METCL(nn.Module):
    def __init__(self, args):
        
        super(METCL, self).__init__()
        
        self.model = Local_Model.from_pretrained(args.text_backbone, cache_dir = args.cache_path, args = args)
        self.cons_model = Cons_Model.from_pretrained(args.text_backbone, cache_dir = args.cache_path, args = args)
        
        self.ctx_vectors = self._init_ctx(args)
        self.ctx = nn.Parameter(self.ctx_vectors)

        self.label_len = args.label_len
        args.feat_size = args.text_feat_dim
        args.video_feat_size = args.video_feat_dim
        args.audio_feat_size = args.audio_feat_dim
    

    def _init_ctx(self, args):
        ctx = torch.empty(args.prompt_len, args.text_feat_dim, dtype=torch.float)
        nn.init.trunc_normal_(ctx)
        return ctx

    
    def forward(self, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):
        video_feats = video_feats.float()
        audio_feats = audio_feats.float()


        outputs, pooled_output, condition, generated_ctx = self.model(
            text = text_feats,
            visual = video_feats,
            acoustic = audio_feats,
            condition_idx=condition_idx, 
            ctx=self.ctx
        )

   
        cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:, 1], cons_text_feats[:, 2]

        cons_outputs = self.cons_model(
            input_ids = cons_input_ids, 
            condition_idx=condition_idx,
            ctx=generated_ctx,
            token_type_ids = cons_segment_ids, 
            attention_mask = cons_input_mask
        )
        last_hidden_state = cons_outputs.last_hidden_state

        cons_condition_tuple = tuple(last_hidden_state[torch.arange(last_hidden_state.shape[0]),\
            condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
 
        cons_condition = torch.cat(cons_condition_tuple, dim=1)


        return outputs[0], pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1)
    


