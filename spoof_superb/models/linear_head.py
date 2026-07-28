import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl.nn import S3PRLUpstream, Featurizer


class SSLModel(nn.Module):
    def __init__(self, n_layerss, device, args):
        super(SSLModel, self).__init__()
        self.device = device
        self.model_name = args.ssl_feature
        self.model = S3PRLUpstream(self.model_name).to(self.device)
        self.featurizer = Featurizer(self.model).to(self.device)
        self.n_layers=n_layerss
        self.out_dim = self.featurizer.output_size

    def extract_feat_featurizer(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        hs, hs_len = self.featurizer(all_hs, all_hs_len)
        return hs, hs_len
    
    def extract_feat(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in all_hs[:self.n_layers]], dim=1)
    
    def _sample_indices(self, total_layers: int):
        k = min(self.n_layers, total_layers)
        if k == total_layers:
            return list(range(total_layers))
        step = (total_layers - 1) / (k - 1)
        return [int(step * i) for i in range(k)]

    def extract_feat_sample(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1)] * waveform.size(0))
        with torch.no_grad():
            all_hs, _ = self.model(waveform.to(self.device), wavs_len.to(self.device))
        # sample your indices
        idxs = self._sample_indices(len(all_hs))
        # print(idxs)
        # pick & permute
        feats = []
        for i in idxs:
            t = all_hs[i]
            x = t[0].permute(1,0,2) if isinstance(t, tuple) else t
            feats.append(x)
        # result: (batch, chosen_layers, time, dim)
        # print(torch.stack(feats, dim=1).shape)
        return torch.stack(feats, dim=1)
    
    def extract_feat_1n(self, waveform):
        # print(waveform.shape,wavs_len.shape)
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        # print(waveform.shape,wavs_len.shape)
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in all_hs[1:self.n_layers + 1]], dim=1)

    def freeze_feature_extraction(self):
        """Freezes the feature extraction layers of the base SSL model."""
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
    

def get_downstream_model(input_dim, output_dim, config):
    model_cls = eval(config['select'])
    model_conf = config.get(config['select'], {})
    model = model_cls(input_dim, output_dim, **model_conf)
    return model


class FrameLevel(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens=None, activation='ReLU', **kwargs):
        super().__init__()
        latest_dim = input_dim
        self.hiddens = []
        if hiddens is not None:
            for dim in hiddens:
                self.hiddens += [
                    nn.Linear(latest_dim, dim),
                    getattr(nn, activation)(),
                ]
                latest_dim = dim
        self.hiddens = nn.Sequential(*self.hiddens)
        self.linear = nn.Linear(latest_dim, output_dim)

    def forward(self, hidden_state, features_len=None):
        hidden_state = self.hiddens(hidden_state)
        logit = self.linear(hidden_state)

        return logit, features_len


class UtteranceLevel(nn.Module):
    def __init__(self,
        args,
        device,
        pooling='MeanPooling',
        activation='ReLU',
        pre_net=None,
        post_net={'select': 'FrameLevel'},
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.args = args
        self.ssl_model = SSLModel(24, device=self.device, args=self.args)
        self.input_dim = self.ssl_model.out_dim
        self.output_dim = 2
        latest_dim = 256

        self.projector = nn.Linear(self.input_dim, latest_dim)

        self.pre_net = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation)
        self.post_net = get_downstream_model(latest_dim, self.output_dim, post_net)

    def forward(self, x, features_len=None):
        if self.pre_net is not None:
            hidden_state, features_len = self.pre_net(hidden_state, features_len)

        x_ssl_feat, features_len = self.ssl_model.extract_feat_featurizer(x)
        hidden_state = self.projector(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        pooled, features_len = self.pooling(hidden_state, features_len)
        logit, features_len = self.post_net(pooled, features_len)

        return logit


class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()


class AttentivePooling(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, input_dim, activation, **kwargs):
        super(AttentivePooling, self).__init__()
        self.sap_layer = AttentivePoolingModule(input_dim, activation)

    def forward(self, feature_BxTxH, features_len):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)

        return sap_vec, torch.ones(len(feature_BxTxH)).long()


class AttentivePoolingModule(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w
