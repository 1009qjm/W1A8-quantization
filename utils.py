import torch

def quantize(x,scale):
    return torch.clamp(torch.round(x*scale),-127,127)

def dequantize(x,scale):
    return x/scale

def fakequantize(x,scale):
    return dequantize(quantize(x,scale),scale)

def paramTransform(gamma,beta,alpha,mean,var,scale,scale_next):
    #gamma(n,) beta(n,) alpha(n,) mean(n,) var(n,) scale(1,) scale(1,)
    gamma=gamma.cuda()
    beta=beta.cuda()
    alpha=alpha.squeeze().cuda()
    mean=mean.cuda()
    var=var.cuda()
    scale=scale.cuda()
    scale_next=scale_next.cuda()
    #print(gamma.size(),beta.size(),alpha.size(),mean.size(),var.size(),scale.size(),scale_next.size())
    return gamma*alpha/(torch.sqrt(var)*scale)*scale_next,(beta-mean*gamma/torch.sqrt(var))*scale_next

def f(gamma,beta,x,flag):            #x.size=b,n,h,w    gamma.size=n
    gamma=gamma.squeeze()
    if flag:
        return torch.clamp(torch.round(gamma.view(1,-1,1,1)*torch.relu(x)+beta.view(1,-1,1,1)),-127,127)
    else:
        return gamma.view(1,-1,1,1)*torch.relu(x)+beta.view(1,-1,1,1)