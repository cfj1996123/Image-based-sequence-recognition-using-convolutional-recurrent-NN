import torch.nn as nn
import torchvision.models as models


class crnn(nn.Module):
    def __init__(self, hid_dim, chardict):
        super().__init__()
        self.chardict = chardict
        self.char_dim = len(chardict)

        ## (batch_size,3,a,b) --> (batch_size,256,a/8,b/8)
        self.vgg16 = models.vgg16_bn(pretrained=False).features[:24]   ## it is important to use batch normalization here to ensure convergence
        #self.vgg16 = models.vgg16(pretrained=False).features[:17]
        ## (seq_len,batch_size,dim1) --> (seq_len,batch_size,2*hid_dim)
        self.bilstm = nn.LSTM(input_size=256 * 4, hidden_size=hid_dim, batch_first=False, num_layers=2,
                              dropout=0, bidirectional=True)
        self.linear1 = nn.Linear(in_features=2 * hid_dim, out_features=self.char_dim+1, bias=False)  # add a non-character class

    def forward(self, x):
        x = self.vgg16(x)
        x = x.permute(3, 0, 1, 2)
        size = x.size()
        z = x.view(size[0], size[1], size[2] * size[3])
        z = self.bilstm(z)[0]
        #print(z.size())
        z = self.linear1(z)  # out: (seq_len,batch_size,char_dim+1)
        #print(z.size())
        return z

    def seq_to_text(self, seq):
        text = []
        for i in range(len(seq)):
            if (i == 0 or seq[i] != seq[i-1]) and seq[i] > 0:
                text.append(self.chardict[seq[i]-1])

        return text


if __name__ == '__main__':
    print(models.vgg16())