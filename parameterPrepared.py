from quantization import *
from utils import *
device = torch.device('cuda:0')

def parameterTransform(param):
        weight_bin = weight_tnn_bin().to(device)
        #一个四个卷积层和四个BN层
        W1=weight_bin(param['quan_model.0.q_conv.weight'])
        s1=param['quan_model.0.q_conv.activation_quantizer.scale']
        gamma1=param['quan_model.0.bn.weight']
        beta1=param['quan_model.0.bn.bias']
        mean1=param['quan_model.0.bn.running_mean']
        var1=param['quan_model.0.bn.running_var']
        alpha1=torch.mean(torch.abs(param['quan_model.0.q_conv.weight']), (3, 2, 1), keepdim=True)

        W2 = weight_bin(param['quan_model.2.q_conv.weight'])
        s2 = param['quan_model.2.q_conv.activation_quantizer.scale']
        gamma2 = param['quan_model.2.bn.weight']
        beta2 = param['quan_model.2.bn.bias']
        mean2 = param['quan_model.2.bn.running_mean']
        var2 = param['quan_model.2.bn.running_var']
        alpha2 = torch.mean(torch.abs(param['quan_model.2.q_conv.weight']), (3, 2, 1), keepdim=True)

        W3 = weight_bin(param['quan_model.3.q_conv.weight'])
        s3 = param['quan_model.3.q_conv.activation_quantizer.scale']
        gamma3 = param['quan_model.3.bn.weight']
        beta3 = param['quan_model.3.bn.bias']
        mean3 = param['quan_model.3.bn.running_mean']
        var3 = param['quan_model.3.bn.running_var']
        alpha3 = torch.mean(torch.abs(param['quan_model.3.q_conv.weight']), (3, 2, 1), keepdim=True)

        W4 = weight_bin(param['quan_model.5.q_conv.weight'])
        s4 = param['quan_model.5.q_conv.activation_quantizer.scale']
        gamma4 = param['quan_model.5.bn.weight']
        beta4 = param['quan_model.5.bn.bias']
        mean4 = param['quan_model.5.bn.running_mean']
        var4 = param['quan_model.5.bn.running_var']
        alpha4 = torch.mean(torch.abs(param['quan_model.5.q_conv.weight']), (3, 2, 1), keepdim=True)

        g0 = gamma1/torch.sqrt(var1)*s1
        b0 = s1*(beta1-mean1*gamma1/torch.sqrt(var1))
        g1, b1 = paramTransform(gamma2, beta2, alpha1, mean2, var2, s1, s2)
        g2, b2 = paramTransform(gamma3, beta3, alpha2, mean3, var3, s2, s3)
        g3, b3 = paramTransform(gamma4, beta4, alpha3, mean4, var4, s3, s4)
        g4 = alpha4 / s4
        b4 = torch.zeros(g4.size())
        g0 = g0.cuda()
        b0 = b0.cuda()
        g1 = g1.cuda()
        b1 = b1.cuda()
        g2 = g2.cuda()
        b2 = b2.cuda()
        g3 = g3.cuda()
        b3 = b3.cuda()
        g4 = g4.cuda()
        b4 = b4.cuda()
        W1 = W1/alpha1
        W2 = W2/alpha2
        W3 = W3/alpha3
        W4 = W4/alpha4

        return W1,W2,W3,W4,g0,g1,g2,g3,g4,b0,b1,b2,b3,b4