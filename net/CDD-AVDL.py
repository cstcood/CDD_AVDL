import os

import torch
from torch import nn

from net.AST import ASTModel
from net.Layerfusion import patch_concat
from net.MLSTM_FCN import MLSTMfcn




class CDD_AVDL(nn.Module):
    def __init__(self,ast_feature_input_num=64,ast_kernel_size=2,ast_feature_out_num=8,MLSTMfcn_out=4,feature_num=2):
        super(CDD_AVDL,self).__init__()
        self.astmodel= ASTModel(ast_feature_out_num,input_tdim=1190,input_fdim=ast_feature_input_num)
        self.Conv1d=nn.Conv1d(768,ast_feature_input_num,ast_kernel_size)#768
        self.mlstmfun=MLSTMfcn(num_classes=MLSTMfcn_out)
        self.FN=nn.Linear(MLSTMfcn_out+ast_feature_out_num,feature_num)
        self.layerNorma1 = nn.LayerNorm(ast_feature_input_num)
        self.layerNorma2=nn.LayerNorm(MLSTMfcn_out+ast_feature_out_num)
        # TODO:parameters should be recomputed for protection and the parament
        #  can run in __main__ given shape
        #The parameters of patch should be computed by seq1 and seq2
        self.pc1=patch_concat([128,7161],[1192,64],32)
        self.pc_cov1 = torch.nn.Conv2d(73, 73 // 2, kernel_size=3, padding=1, stride=1)
        self.relu1=torch.nn.ReLU()
        self.pc2=patch_concat([64,7157],[32,32],32,chanel2=73//2)
        self.pc_cov2 = torch.nn.Conv2d(147, 147// 2, kernel_size=3, padding=1,
                                       stride=1)
        self.relu2 = torch.nn.ReLU()
        self.pc3 = patch_concat([32, 7155], [32,32], 8,chanel2=147//2)
        self.pc_cov3 = torch.nn.Conv2d(296, 296 // 2, kernel_size=3, padding=1,
                                       stride=1)
        self.relu3 = torch.nn.ReLU()
        self.pc_cov4 = torch.nn.Conv2d(148, 8, kernel_size=3, padding=1,
                                       stride=1)
        self.relu4=torch.nn.ReLU()
        self.FN_pc = torch.nn.Linear(512,2)
    def forward(self,x_audio,x_video,seq_video):

        # input:
        # x_audio feature extracted from wav2vec2 [B,seq1,768]
        # x_video feature extracted from OpenGraphAU [B,seq2,41] seq_video [B]
        # ouput: [B,2]
        x_1=x_audio
        x_2=x_video
        seq_2=seq_video
        x_1= torch.transpose(x_1.float(),1,2)
        x_1= self.Conv1d(x_1)
        x_1= torch.transpose(x_1, 1, 2)
        x_1_0=self.layerNorma1(x_1)

        x_1= self.astmodel(x_1_0)
        out_2,[conv1,conv2,conv3]=self.mlstmfun(x_2,seq_2)


        y1,q1,q2=self.pc1(conv1,x_1_0)
        y1_p=self.relu1(self.pc_cov1(y1))
        y2,q1,q2 =self.pc2(conv2, y1_p)
        y2_p = self.relu2(self.pc_cov2(y2))
        y3,p1,p2 = self.pc3(conv3, y2_p)
        y3_p =  self.relu3(self.pc_cov3(y3))
        y3_p = self.relu3(self.pc_cov4(y3_p))
        y3= torch.flatten(y3_p,1)
        y3= self.FN_pc(y3)


        x_all = torch.cat((x_1, out_2), dim=1)
        x_all= self.layerNorma2(x_all)
        y_tag=self.FN(x_all)
        y_tag=nn.functional.log_softmax((y3 + y_tag) / 2, dim=1)
        return y_tag
if __name__ == '__main__':
    os.environ['TORCH_HOME'] = './pretrained_models'
    audio_x=torch.randn([8,1193,768])
    video_x=torch.randn([8,7168,41])
    seq_x=torch.randint(6860,7168,size=(8,))
    cdd_avdl= CDD_AVDL()
    y=cdd_avdl(audio_x,video_x,seq_x)
    print(y.shape)