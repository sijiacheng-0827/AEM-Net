import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mer_spot.lib.config import cfg, update_config

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))  # 平滑版的RELU函数
        return x



############### Postbackbone ##############
class BaseFeatureNet(nn.Module): # 主干网络
    '''
    Calculate basic feature
    PreBackbobn -> Backbone
    CAS(ME)^2:
    input: [batch_size, 2048, 64]
    output: [batch_size, 512, 16]
    SAMM:
    input: [batch_size, 2048, 256]
    output: [batch_size, 512, 64]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.dataset = cfg.DATASET.DATASET_NAME

        # #############################################


        self.conv3 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,  # IN_FEAT_DIM: 2048
                               out_channels=384,  # BASE_FEAT_DIM: 512
                               kernel_size=5, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=2048,  # 512
                               out_channels=640,  # 512
                               kernel_size=5, stride=1, padding=1, bias=True)

        self.conv5 = nn.Conv1d(in_channels=2048,  # 512
                               out_channels=1024,  # 512
                               kernel_size=5, stride=1, padding=1, bias=True)




        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,  # IN_FEAT_DIM: 2048
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,  # BASE_FEAT_DIM: 512
                               kernel_size=3, stride=2, padding=2 , bias=True)
        # self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,  # 512
        #                        out_channels=cfg.MODEL.BASE_FEAT_DIM,  # 512
        #                        kernel_size=7, stride=1, padding=4, bias=True)#原始k=9
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.max_pooling1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mish = Mish()  # x = x * (torch.tanh(F.softplus(x)))   平滑版的RELU函数
        # self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(384)
        self.bn2 = nn.BatchNorm1d(640)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
    def forward(self, x):
        feat1 = self.mish(self.bn1(self.conv3(x)))
        # feat = self.relu(self.conv1(x))
        # feat1 = self.max_pooling1(feat1)
        feat2 = self.mish(self.bn2(self.conv4(x)))
        # feat2 = self.relu(self.conv2(x))
        # feat2 = self.max_pooling1(feat2)
        feat3 = self.mish(self.bn3(self.conv5(x)))
        # feat = self.relu(self.conv1(x))
        # feat3 = self.max_pooling1(feat3)
        merged = torch.cat((feat1, feat2, feat3),dim=1)
        # merged_pool = self.max_pooling(merged)
################################################
        feat = self.mish(self.bn4(self.conv1(merged)))
        # feat = self.mish(self.conv1(x))#原始代码

        feat = self.max_pooling(feat)


    #
    #     self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM, out_channels=cfg.MODEL.BASE_FEAT_DIM,
    #          kernel_size=3, stride=1, padding=1, bias=True)
    #     self.conv2  = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM, out_channels=cfg.MODEL.IN_FEAT_DIM,
    #                                         kernel_size=3, stride=1, padding=1, bias=True)
    #     self.conv3 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
    #                                         out_channels=cfg.MODEL.BASE_FEAT_DIM,
    #                                         kernel_size=7, stride=2, padding=2, bias=True)
    #     self.conv4 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
    #                                         out_channels=cfg.MODEL.BASE_FEAT_DIM,
    #                                         kernel_size=5, stride=2, padding=2, bias=True)
    #     self.bn = nn.BatchNorm1d(512)
    #     self.bn1 = nn.BatchNorm1d(2048)
    #     self.mish = Mish()
    #     self.head = 4
    #     self.query = nn.Conv1d(512, 512, kernel_size=1)
    #     self.key = nn.Conv1d(512, 512, kernel_size=1)
    #     self.value = nn.Conv1d(512, 512, kernel_size=1)
    #     self.rel_len = nn.Parameter(torch.randn([1, 4, 128, 64]), requires_grad=True) # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
    #     # self.rel_len = torch.randn(1, 512, 1, 64)
    #     self.softmax = nn.Softmax(dim=-1)
    #
    # def forward(self, x):
    #     identity = x
    #     x1 = self.mish(self.bn(self.conv1(x)))
    #     n_batch, c, l = x1.size()
    #     q = self.query(x1).view(n_batch, 4, -1)
    #     k = self.key(x1).view(n_batch, 4, -1)
    #     v = self.value(x1).view(n_batch, 4, -1)
    #     content_content = torch.matmul(q.permute(0, 2, 1), k)
    #     content_position = self.rel_len.view(1, c, -1).permute(0, 2, 1)
    #     content_position = torch.matmul(content_position, q)
    #     energy = content_content + content_position
    #     attention = self.softmax(energy)
    #     feat_mid = torch.matmul(v, attention.permute(0, 2, 1))
    #     feat_mid = feat_mid.view(n_batch, c, l)
    #     feat_l = self.mish(self.bn1(self.conv2(feat_mid)))
    #     feat_m = feat_l + identity
    #     feat = self.mish(self.bn(self.conv3(feat_m)))
    #     feat = self.mish(self.bn(self.conv4(feat)))

        return feat


############### Neck ##############
class FeatNet(nn.Module):
    '''
    Main network
    Backbone -> Neck
    CAS(ME)^2:
    input: base feature, [batch_size, 512, 16]
    output: MAL1, MAL2, MAL3, MAL4
    MAL1: [batch_size, 512, 16]
    MAL2: [batch_size, 512, 8]
    MAL3: [batch_size, 1024, 4]
    MAL4: [batch_size, 1024, 2]
    SAMM:
    input: base feature, [batch_size, 512, 128]
    output: MAL1, MAL2, MAL3, MAL4, MAL5, MAL6, MAL7
    MAL1: [batch_size, 1024, 32]
    MAL2: [batch_size, 1024, 16]
    MAL3: [batch_size, 1024, 8]
    MAL4: [batch_size, 1024, 4]
    MAL5: [batch_size, 1024, 2]
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_feature_net = BaseFeatureNet(cfg)  # conv1，conv2 kernel_size = 9
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):  # NUM_LAYERS: 4
            # stride = 1 if layer == 0 else 2
            in_channel = cfg.MODEL.BASE_FEAT_DIM if layer == 0 else cfg.MODEL.LAYER_DIMS[layer - 1]  # LAYER_DIMS: [512, 512, 1024, 1024]
            out_channel = cfg.MODEL.LAYER_DIMS[layer]  # LAYER_DIMS: [512, 512, 1024, 1024]
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[layer], padding=1)
            # LAYER_STRIDES: [1, 2, 2, 2]
            self.convs.append(conv)  # 循环一次加一个卷积层
        # self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()  # x = x * (torch.tanh(F.softplus(x))) 平滑版的RELU函数
        self.up_feat = nn.Upsample(scale_factor=2)  # 我写的


    def forward(self, x):  # 作者的
        results1 = []
        feat = self.base_feature_net(x)
        for conv in self.convs:
            feat = self.mish(conv(feat))
            # self.up_feat = nn.Upsample(scale_factor=2)
            # feat_up = self.up_feat(feat)
            # feat[1] = feat_up[2] +feat_up[1]
            # print(feat.shape)
            results1.append(feat)  # base_feature_net = BaseFeatureNet(cfg)   conv1，conv2 kernel_size = 9

        results = []
        results.append(results1[3])  # conv4特征
        feat_up_3 = self.up_feat(results1[3])  # conv4上采样特征
        feat_3 = feat_up_3 + results1[2]  # 融合后conv3特征
        results.append(feat_3)  # conv3特征
        feat_up_2 = self.up_feat(feat_3)  # 融合后conv3上采样
        feat_2 = feat_up_2 + results1[1]  # 融合后conv2特征
        results.append(feat_2)  # conv2特征
        feat_up_1 = self.up_feat(feat_2)
        feat_1 = feat_up_1 + results1[0]
        results.append(feat_1)
        results.reverse()
        return tuple(results)


# Postbackbone -> Neck
class GlobalLocalBlock(nn.Module):  #Location supression modules
    def __init__(self, cfg):
        super(GlobalLocalBlock, self).__init__()


        self.dim_in = cfg.MODEL.BASE_FEAT_DIM
        self.dim_out = cfg.MODEL.BASE_FEAT_DIM  #  BASE_FEAT_DIM: 512
        self.ws = cfg.DATASET.WINDOW_SIZE  # WINDOW_SIZE: 128
        self.drop_threshold = cfg.MODEL.DROP_THRESHOLD  # DROP_THRESHOLD: 0.45
        self.ss = cfg.DATASET.SAMPLE_STRIDE # SAMPLE_STRIDE: 2
        self.mish = Mish()
        
        self.down = nn.Conv1d(self.dim_in, self.dim_out//2, kernel_size=1, stride=1)
        self.theta = nn.Conv1d(self.dim_in//2, self.dim_out//2, kernel_size=1, stride=1)
        self.phi = nn.Conv1d(self.dim_in//2, self.dim_out//2, kernel_size=1, stride=1)
        self.g = nn.Conv1d(self.dim_in//2, self.dim_out//2, kernel_size=1, stride=1)

        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                     nn.Conv1d(self.dim_in//2,self.dim_in//32,kernel_size=1, stride=1),
                                     nn.ReLU(),
                                     nn.Conv1d(self.dim_in//32,self.dim_in//2,kernel_size=1, stride=1),
                                     nn.Sigmoid())
        #Fuse
        self.lcoal_global = nn.Conv1d(self.dim_out//2, self.dim_out//2, kernel_size=1, stride=1)

        # MLP
        # self.drop = nn.Dropout(p=0.3)
        self.drop = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(self.dim_out//2, 4*self.dim_out//2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(4*self.dim_out//2, self.dim_out//2, kernel_size=1, stride=1)
        self.up = nn.Conv1d(self.dim_out//2, self.dim_out, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x= self.down(x)  # 下采样
        residual = x

        batch_size = x.shape[0]
        channels = x.shape[1]
        ori_length = x.shape[2]
        
        length_temp = self.ws //(self.ss*ori_length)  #  batch_size = x.shape[0],channels = x.shape[1] ori_length = x.shape[2]
        # WINDOW_SIZE: 128  # 特征图数量
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        # all_tmp = torch.zeros([channels, batch_size, length_temp, ori_length]).cuda()
        all_tmp = torch.zeros([channels, batch_size, length_temp, ori_length]).cuda()
        all_temp_g = all_tmp
        for j in range(theta.size(1)):# 256
            # Sometimes, temp1: BS* T * Channels
            # temp2: BS* (T+1) * Channels
            # temp = torch.zeros([batch_size, length_temp, ori_length]).cuda() # BS*T*mean_channel length_temp=ws //(ss*ori_length) ss= 2
            temp = torch.zeros([batch_size, length_temp, ori_length]).cuda()
            temp_g = temp
            if j < length_temp//2:
                temp[:,length_temp//2-j:,:] = theta[:,:j+length_temp//2,:]
                temp_g[:,length_temp//2-j:,:] = g[:,:j+length_temp//2,:]
            elif length_temp//2 <= j <= theta.size(1)-length_temp//2:
                temp = theta[:,j-length_temp//2:j+length_temp//2,:]
                temp_g= g[:,j-length_temp//2:j+length_temp//2,:]
            else:
                temp[:,:length_temp-(j%length_temp-length_temp//2),:] = theta[:,j-length_temp//2:,:]
                temp_g[:,:length_temp-(j%length_temp-length_temp//2),:] = g[:,j-length_temp//2:,:]
            
            all_tmp[j:j+1,:,:,:]= temp
            all_temp_g[j:j+1,:,:,:] = temp_g


        # phi_se = self.se(phi)
        # phi_at = phi_se * phi
        # # phi_in = phi_at + phi
        # #
        # all_tmp_phi = phi_at.unsqueeze(dim=2)
        all_tmp_phi = phi.unsqueeze(dim=2)

        local_theta_phi = torch.matmul(all_tmp_phi, all_tmp.permute(1,0,3,2))  # 计算两个矩阵乘积


        local_theta_phi_sc = local_theta_phi * (channels**-.5)
        local_p = F.softmax(local_theta_phi_sc, dim=-1)
        # local_p = F.softmax(local_theta_phi, dim=-1)
        local_p = local_p.expand(-1, -1, ori_length, -1)
        all_temp_g = all_temp_g.permute(1,0,3,2)
        # all_temp_g = torch.where(all_temp_g < torch.tensor(self.drop_threshold).float().cuda(), torch.tensor(0).float().cuda(), all_temp_g)
        all_temp_g = torch.where(all_temp_g < torch.tensor(self.drop_threshold).float().cuda(),
                                 torch.tensor(0).float().cuda(), all_temp_g)
        local_temp = torch.sum(self.drop(local_p) * all_temp_g, dim=-1)
        out_temp = local_temp
    
        # global temporal encoder
        # e.g. (BS, 512, 16) * (BS, 16, 512) => (BS, 1024, 1024)
        # global_theta_phi = torch.bmm(phi, torch.transpose(theta,2,1))
        # global_theta_phi_sc = global_theta_phi * (channels**-.5)
        # global_p = F.softmax(global_theta_phi_sc, dim=-1)
        # global_temp = torch.bmm(self.drop(global_p), g)

        # out_temp = torch.cat((local_temp, global_temp), dim=1)

        # MLP
        local_global = self.lcoal_global(out_temp)
        out_temp_ln = F.layer_norm(self.drop(local_global)+residual,[channels, ori_length])  ##############残差
        
        out_mlp_conv1 = self.conv1(out_temp_ln)
        out_mlp_act = self.mish(self.drop(out_mlp_conv1))
        out_mlp_conv2 = self.conv2(out_mlp_act)
        out = F.layer_norm(self.drop(out_mlp_conv2) + out_temp_ln, [channels, ori_length])

        out = self.up(out)
        # add
        # out_t = out.cpu()
        # out_t = out_t.detach().numpy()  # (32, 512, 16)
        # out_arr = np.array([sum(out_t[i]) for i in range(32)])
        # # print(out_arr.shape)
        # out_arr = torch.from_numpy(out_arr)

        return out


############### Postneck ##############
class ReduceChannel(nn.Module): #
    '''
    From FeatNet
    Neck -> Postneck
    CAS(ME)^2:
    input: from FeatNet
           MAL1: [batch_size, 512, 16]
           MAL2: [batch_size, 512, 8]
           MAL3: [batch_size, 1024, 4]
           MAL4: [batch_size, 1024, 2]
    output: All Level-features'Channels Reduced into 512
    SAMM:
    input: from FeatNet
           MAL1: [batch_size, 512, 128]
           MAL2: [batch_size, 512, 64]
           MAL3: [batch_size, 1024, 32]
           MAL4: [batch_size, 1024, 16]
           MAL5: [batch_size, 1024, 8]
           MAL6: [batch_size, 1024, 4]
           MAL7: [batch_size, 1024, 2]
    output: All Level-features'Channels Reduced into 512
    '''
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):  # NUM_LAYERS: 4
            conv = nn.Conv1d(cfg.MODEL.LAYER_DIMS[layer], cfg.MODEL.REDU_CHA_DIM, kernel_size=1)  # LAYER_DIMS: [512, 512, 1024, 1024]
              # REDU_CHA_DIM: 512  ???输出为512，下一层输入为什么是1024
            self.convs.append(conv)
        # self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()

    def forward(self, feat_list):
        assert len(feat_list) == len(self.convs)
        results = []
        for conv, feat in zip(self.convs, feat_list):
           results.append(self.mish(conv(feat)))
           # results.append(self.relu(conv(feat)))
        return tuple(results)



############### Head ##############
class PredHeadBranch(nn.Module):  # 输出RES CLS
    '''
    From ReduceChannel Module
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    output: Channels reduced into 256
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    output: Channels reduced into 256
    '''
    def __init__(self, cfg):
        super(PredHeadBranch, self).__init__()
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS  # 2
        self._init_head(cfg)

    def _init_head(self, cfg):
        self.convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.REDU_CHA_DIM if layer == 0 else cfg.MODEL.HEAD_DIM
            out_channel = cfg.MODEL.HEAD_DIM
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.convs.append(conv)
        self.mish = Mish()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = x
        for conv in self.convs:
            feat = self.mish(conv(feat))
            # feat = self.relu(conv(feat))
        return feat


############### Details of Prediction ##############
class PredHead(nn.Module):
    '''
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    input_tmp: to PredHeadBranch Module
    output: Channels reduced into number of classes or boundaries
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    input_tmp: to PredHeadBranch Module
    output: Channels reduced into number of classes or boundaries
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.head_branches = nn.ModuleList()
        self.lgf= GlobalLocalBlock(cfg)
        self.inhibition = cfg.MODEL.INHIBITION_INTERVAL# 【16，2】
        for _ in range(4):
            self.head_branches.append(PredHeadBranch(cfg))
        num_class = cfg.DATASET.NUM_CLASSES  # 2
        num_box = len(cfg.MODEL.ASPECT_RATIOS)  # 5

        # [batch_size, 256, (16,8,4,2)] -> [batch_size, _, (16,8,4,2)]
        af_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_class, kernel_size=3, padding=1)#HEAD_DIM = 256 num_class = 2
        af_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, 2, kernel_size=3, padding=1)
        ab_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * num_class, kernel_size=3, padding=1)
        ab_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * 2, kernel_size=3, padding=1)#num_box = 5
        self.pred_heads = nn.ModuleList([af_cls, af_reg, ab_cls, ab_reg])

    def forward(self, x):
        preds = []
        if x.size(-1) in self.inhibition:#[16, 2]
            lgf_out = self.lgf(x)
        else:
            lgf_out = x
        for pred_branch, pred_head in zip(self.head_branches, self.pred_heads):
            feat = pred_branch(lgf_out)
            preds.append(pred_head(feat))

        return tuple(preds)


############### Prediction ##############
class LocNet(nn.Module): # nonlocal
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        # self.features = FeatNet(cfg)
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = PredHead(cfg)
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.ab_pred_value = cfg.DATASET.NUM_CLASSES + 2 #   NUM_CLASSES: 2

    def _layer_cal(self, feat_list):
        af_cls = list()
        af_reg = list()
        ab_pred = list()

        for feat in feat_list:
            cls_af, reg_af, cls_ab, reg_ab = self.pred(feat)
            af_cls.append(cls_af.permute(0, 2, 1).contiguous())
            af_reg.append(reg_af.permute(0, 2, 1).contiguous())
            ab_pred.append(self.tensor_view(cls_ab, reg_ab))

        af_cls = torch.cat(af_cls, dim=1)  # bs, sum(t_i), n_class+1
        af_reg = torch.cat(af_reg, dim=1)  # bs, sum(t_i), 2
        af_reg = F.relu(af_reg)
        return (af_cls, af_reg), tuple(ab_pred)

    def tensor_view(self, cls, reg):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        bs, c, t = cls.size()
        cls = cls.view(bs, -1, self.num_class, t).permute(0, 3, 1, 2).contiguous()
        reg = reg.view(bs, -1, 2, t).permute(0, 3, 1, 2).contiguous()
        data = torch.cat((cls, reg), dim=-1)
        data = data.view(bs, -1, self.ab_pred_value)
        return data

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)

        return self._layer_cal(features_list)

############### All processing ##############
class FuseModel(nn.Module):
    def __init__(self, cfg):
        super(FuseModel, self).__init__()
        self.features = FeatNet(cfg)
        self.loc_net = LocNet(cfg)

    def forward(self, x):
        features = self.features(x)
        out_af, out_ab = self.loc_net(features)
        return out_af, out_ab
#######out_af = 2*8*30  out_ab = 4*8*80
if __name__ == '__main__':
    import sys
    sys.path.append('D:/codeassist/lssnet/mer_spot/lib')

    cfg_file = 'D:/codeassist/lssnet/mer_spot/experiments/CAS.yaml'
    #     sys.path.append('/home/yww/1_spot/MSA-Net/lib')
    #     cfg_file = '/home/yww/1_spot/MSA-Net/experiments/A2Net_thumos.yaml'
    update_config(cfg_file)

    model = FuseModel(cfg).cuda()


    data = torch.randn((8, 2048, 64)).cuda()
    output = model(data)


# #########practice#######
# #practice
class Basicblock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    def forword(self, x):
         identity = x
         out = self.conv1(x)
         out = self.bn1(out)
         out = self.relu(out)
         out = self.conv2(out)
         out += identity
         out = self.relu(out)
         return out

#
class Resnet(nn.Module):
    def __init__(self, block, block_num, num_classes=3, include_top = True):  #include_top:在resnet的基础上搭建更复杂的网络
        super(Resnet, self).__init__()
        include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, block_num[0])
        self.layer2 = self.make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self.make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self.make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)


    def make_layer(self, block, channel, block_num, stride=1):
        downsample = None  # 50, 101, 152 将图片高宽缩减为原来的一半
        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel
        for i in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out